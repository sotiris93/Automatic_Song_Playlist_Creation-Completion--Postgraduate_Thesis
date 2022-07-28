class Playlist:
    """
    A Playlist object.
    """
    def __init__(self, playlist: dict):
        self._playlist = playlist

    def uris(self, uri_type: str):
        """ A list containing the URIs corresponding to the specified type """
        return [track[uri_type] for track in self._playlist['tracks']]

    def album_uris(self):
        """ A list holding the album URIs """
        return self.uris('album_uri')

    def artist_uris(self):
        """ A list holding the artist URIs """
        return self.uris('artist_uri')

    def track_uris(self):
        """ A list holding the track URIs """
        return self.uris('track_uri')

    def name(self):
        """ Returns the name of the playlist """
        try:
            return self._playlist['name']
        except KeyError:
            return "No Name"

    def pid(self):
        """ Returns the playlist ID """
        return self._playlist['pid']
