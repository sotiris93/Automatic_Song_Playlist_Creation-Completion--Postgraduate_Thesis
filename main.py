""" This is the main script that runs all the calculations """
from typing import Tuple
import pickle
from joblib import Parallel, delayed
import numpy as np
import json
import os
import re
import csv
from pathlib import Path
from tqdm import tqdm
from playlist import Playlist


def jaccard_similarity(A, B):
    """ Computes the Jaccard similarity between two iterables """
    inter = set(A).intersection(B)
    join = set(A).union(B)
    return len(inter) / len(join)


def jaccard_score(p: Playlist, q: Playlist) -> float:
    """
    Calculates the Jaccard score
    """
    if p.name() is None or q.name() is None:
        denominator = 1
    else:
        denominator = 1 + jaccard_similarity(normalize_name(p.name()),
                                             normalize_name(q.name()))

    jaccard_track = jaccard_similarity(p.track_uris(), q.track_uris())
    jaccard_artist = jaccard_similarity(p.artist_uris(), q.artist_uris())
    jaccard_album = jaccard_similarity(p.album_uris(), q.album_uris())

    return (1 + 0.5 * jaccard_track + 0.25 * jaccard_artist +
            0.25 *
            jaccard_album) / denominator


def normalize_name(name):
    """
    Cleans up a name by removing special characters and multiple whitespace
    """
    name = name.lower()
    name = re.sub(r"[.,/#!$%^*;:{}=_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def load_playlists(path):
    """ Loads the playlists from a playlist file """
    with open(path) as fp:
        pls = [Playlist(playlist) for playlist in json.load(fp)[
            'playlists']]
    return pls


def sort_playlists(playlists, ratings):
    """ Sorts the playlists by their ratings """
    ratings = np.array(ratings)
    playlists = np.array(playlists)
    playlists = playlists[np.argsort(ratings)][::-1]  # argsort sorts
    # ascendingly so reverses the order
    return playlists, np.flip(np.sort(ratings))


def get_recommendations(playlists: list, ratings: list, seed_tracks: list,
                        n_tracks: int) -> list:
    """ Creates a playlist with n_tracks from the highest rated playlists """
    seed_tracks = set(seed_tracks)
    playlists, ratings = sort_playlists(playlists, ratings)
    recommendations = set()
    for playlist in playlists:
        if len(recommendations) < n_tracks:
            recommendations = recommendations.union(set(playlist.track_uris()).difference(
                seed_tracks))
        else:
            break
    return list(recommendations)[:n_tracks]


def load_checkpoint(pl_name):
    """
    Loads a checkpoint file containing the 100 top-rated playlists
    determined from the listed mpd_slice_files

    pl_name: Name of the playlist to load the checkpoint for
    """
    with open(f"./checkpoints-1000-slices/playlist_{pl_name}_top100", 'rb') as f:
        mpd_slice_files, top100_playlists, top100_ratings = pickle.load(f)
    return mpd_slice_files, top100_playlists, top100_ratings

def save_checkpoint(playlist, all_playlists, ratings, mpd_slice_files):
    """
    Saves a checkpoint file containing the 100 top rated playlists
    determined for the listed mpd_slice_file
    """
    checkpoint_file = f"./checkpoints-1000-slices/playlist_{playlist}_top100"
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump([mpd_slice_files, all_playlists, ratings], f,
                    pickle.HIGHEST_PROTOCOL)


def process_playlist(pl) -> Tuple[int, list]:
    """

    pl: The playlist that we want to find recommended tracks for
    returns: A tuple containing the ID of the playlist and a list of 500
    recommended tracks
    """

    ratings = list()
    all_playlists = list()
    mpd_slice_files = list()
    try:
        mpd_slice_files, all_playlists, ratings = load_checkpoint(pl.pid())
    except FileNotFoundError:
        pass
    for mpd_slice_file in Path(r'C:\Users\sotiris\spotify_million_playlist_dataset\data').glob(
            'mpd.slice.*.json'):
        if mpd_slice_file.stem in mpd_slice_files:
            continue
        mpd_playlists = load_playlists(mpd_slice_file)
        for mpd_playlist in mpd_playlists:
            ratings.append(jaccard_score(mpd_playlist, pl))
            all_playlists.append(mpd_playlist)
        # Keep only the top 100 rated playlists
        all_playlists, ratings = sort_playlists(all_playlists, ratings)
        all_playlists = list(all_playlists[:100])
        ratings = list(ratings[:100])
        mpd_slice_files.append(mpd_slice_file.stem)
        save_checkpoint(pl.pid(), all_playlists, ratings,
                        mpd_slice_files)
    return (pl.pid(),
            get_recommendations(all_playlists, ratings, pl.track_uris(),
                                500))


if __name__ == "__main__":
    
    challenge_file_path = r'C:\Users\sotiris\challenge_set.json'
    output_file_path = r'C:\Users\sotiris\Downloads\final_submission.csv'

    # Load the challenge playlists
    challenge_playlists = load_playlists(challenge_file_path)

    # Write header to output file
    header = ['team_info', 'name', 'email']
    with open(output_file_path, 'w', newline='') as csvfile:
        fw = csv.writer(csvfile, delimiter=',')
        fw.writerow(header)



    results = Parallel(n_jobs=-1)(
        delayed(process_playlist)(challenge_playlist)
        for challenge_playlist in tqdm(challenge_playlists)
    )

    with open(output_file_path, 'a',
              newline='') as csvfile:
        for result in results:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow([result[0]] + result[1])

    print("Done")
