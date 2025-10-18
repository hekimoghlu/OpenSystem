/*
 * Copyright (C) 2013-2014 Jolla Ltd. <andrew.den.exter@jollamobile.com>
 * Author: Andrew den Exter <andrew.den.exter@jollamobile.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */


#include <glib.h>

#include <tracker-common.h>

#include <libavformat/avformat.h>

#include "tracker-cue-sheet.h"

#include "utils/tracker-extract.h"

#ifdef HAVE_GUPNP_DLNA
#include "tracker-gupnp.h"
#endif

#define CHUNK_N_BYTES (2 << 15)

static guint64
extract_gibest_hash (GFile *file)
{
	guint64 buffer[2][CHUNK_N_BYTES/8];
	g_autoptr (GInputStream) stream = NULL;
	g_autoptr (GError) error = NULL;
	gssize n_bytes, file_size;
	guint64 hash = 0;
	gint i;

	stream = G_INPUT_STREAM (g_file_read (file, NULL, &error));
	if (stream == NULL)
		goto fail;

	/* Extract start/end chunks of the file */
	n_bytes = g_input_stream_read (stream, buffer[0], CHUNK_N_BYTES, NULL, &error);
	if (n_bytes == -1)
		goto fail;

	if (!g_seekable_seek (G_SEEKABLE (stream), -CHUNK_N_BYTES, G_SEEK_END, NULL, &error)) {
		/* Files may be smaller than the chunk size */
		if (!g_error_matches (error, G_IO_ERROR, G_IO_ERROR_INVALID_ARGUMENT))
			goto fail;

		g_clear_error (&error);
		if (!g_seekable_seek (G_SEEKABLE (stream), 0, G_SEEK_SET, NULL, &error))
			goto fail;
	}

	n_bytes = g_input_stream_read (stream, buffer[1], CHUNK_N_BYTES, NULL, &error);
	if (n_bytes == -1)
		goto fail;

	for (i = 0; i < G_N_ELEMENTS (buffer[0]); i++)
		hash += buffer[0][i] + buffer[1][i];

	file_size = g_seekable_tell (G_SEEKABLE (stream));

	if (file_size < CHUNK_N_BYTES)
		goto end;

	/* Include file size */
	hash += file_size;

	return hash;

 fail:
	g_warning ("Could not get file hash: %s\n", error->message);
 end:
	return 0;
}

static void
add_hash (TrackerResource *resource,
          GFile           *file,
          const char      *hash_str,
          const char      *algorithm)
{
	g_autoptr (TrackerResource) file_resource = NULL, hash = NULL;
	g_autofree char *uri = NULL;

	g_set_object (&file_resource, tracker_resource_get_first_relation (resource, "nie:isStoredAs"));

	if (!file_resource) {
		uri = g_file_get_uri (file);
		file_resource = tracker_resource_new (uri);
		tracker_resource_set_relation (resource, "nie:isStoredAs", file_resource);
	}

	hash = tracker_resource_new (NULL);
	tracker_resource_set_uri (hash, "rdf:type", "nfo:FileHash");
	tracker_resource_set_string (hash, "nfo:hashValue", hash_str);
	tracker_resource_set_string (hash, "nfo:hashAlgorithm", algorithm);

	tracker_resource_set_relation (file_resource, "nfo:hasHash", hash);
}

static void
add_external_reference (TrackerResource *resource,
                        const char      *uri_prefix,
                        const char      *id,
                        const char      *reference_id)
{
	g_autoptr (TrackerResource) reference = NULL;
	g_autofree char *uri = NULL;

	uri = g_strdup_printf ("%s/%s", uri_prefix, id);
	reference = tracker_extract_new_external_reference (reference_id, id, uri);
	tracker_resource_add_relation (resource, "tracker:hasExternalReference", reference);
}

static AVDictionaryEntry *
find_tag (AVFormatContext *format,
          AVStream        *stream1,
          AVStream        *stream2,
          const gchar     *name)
{
	AVDictionaryEntry *tag = av_dict_get (format->metadata, name, NULL, 0);
	if (!tag && stream1) {
		tag = av_dict_get (stream1->metadata, name, NULL, 0);
	}
	if (!tag && stream2) {
		tag = av_dict_get (stream2->metadata, name, NULL, 0);
	}

	return tag;
}

G_MODULE_EXPORT gboolean
tracker_extract_get_metadata (TrackerExtractInfo  *info,
                              GError             **error)
{
	GFile *file;
	g_autoptr (TrackerResource) metadata = NULL;
	g_autofree char *absolute_file_path = NULL;
	g_autofree char *content_created = NULL;
	g_autofree char *uri = NULL, *resource_uri = NULL;
	AVFormatContext *format = NULL;
	AVStream *audio_stream = NULL;
	AVStream *video_stream = NULL;
	int audio_stream_index;
	int video_stream_index;
	AVDictionaryEntry *tag = NULL;
	const char *title = NULL;
	AVDictionary *options = NULL;
#ifdef HAVE_GUPNP_DLNA
	g_autoptr (GUPnPDLNAInformation) gupnp_info = NULL;
	g_autoptr (GUPnPDLNAProfileGuesser) profile_guesser = NULL;
	GUPnPDLNAProfile *gupnp_profile = NULL;
#endif

	file = tracker_extract_info_get_file (info);

	uri = g_file_get_uri (file);

	absolute_file_path = g_file_get_path (file);
	av_dict_set_int (&options, "export_xmp", 1, 0);

	if (avformat_open_input (&format, absolute_file_path, NULL, &options)) {
		av_dict_free (&options);
		return FALSE;
	}

	av_dict_free (&options);
	avformat_find_stream_info (format, NULL);

	audio_stream_index = av_find_best_stream (format, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
	if (audio_stream_index >= 0) {
		audio_stream = format->streams[audio_stream_index];
	}

	video_stream_index = av_find_best_stream (format, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	if (video_stream_index >= 0) {
		video_stream = format->streams[video_stream_index];
	}

	if (!audio_stream && !video_stream) {
		avformat_close_input (&format);
		return FALSE;
	}

	resource_uri = tracker_extract_info_get_content_id (info, NULL);
	metadata = tracker_resource_new (resource_uri);

	if (audio_stream) {
		if (audio_stream->codecpar->sample_rate > 0) {
			tracker_resource_set_int64 (metadata, "nfo:sampleRate", audio_stream->codecpar->sample_rate);
		}
		if (audio_stream->codecpar->ch_layout.nb_channels > 0) {
			tracker_resource_set_int64 (metadata, "nfo:channels", audio_stream->codecpar->ch_layout.nb_channels);
		}
	}

	if (video_stream && !(video_stream->disposition & AV_DISPOSITION_ATTACHED_PIC)) {
		guint64 hash;

		tracker_resource_add_uri(metadata, "rdf:type", "nmm:Video");

		if (video_stream->codecpar->width > 0 && video_stream->codecpar->height > 0) {
			tracker_resource_set_int64 (metadata, "nfo:width", video_stream->codecpar->width);
			tracker_resource_set_int64 (metadata, "nfo:height", video_stream->codecpar->height);
		}

		if (video_stream->avg_frame_rate.num > 0) {
			gdouble frame_rate = (gdouble) video_stream->avg_frame_rate.num
			                     / video_stream->avg_frame_rate.den;
			tracker_resource_set_double (metadata, "nfo:frameRate", frame_rate);
		}

		if (video_stream->duration > 0) {
			gint64 duration = av_rescale(video_stream->duration, video_stream->time_base.num,
			                             video_stream->time_base.den);
			tracker_resource_set_int64 (metadata, "nfo:duration", duration);
		}

		if (video_stream->sample_aspect_ratio.num > 0) {
			gdouble aspect_ratio = (gdouble) video_stream->sample_aspect_ratio.num
			                       / video_stream->sample_aspect_ratio.den;
			tracker_resource_set_double (metadata, "nfo:aspectRatio", aspect_ratio);
		}

		if (video_stream->nb_frames > 0) {
			tracker_resource_set_int64 (metadata, "nfo:frameCount", video_stream->nb_frames);
		}

		if ((tag = find_tag (format, video_stream, NULL, "synopsis"))) {
			tracker_resource_set_string (metadata, "nmm:synopsis", tag->value);
		}

		if ((tag = find_tag (format, video_stream, NULL, "episode_sort"))) {
			tracker_resource_set_int64 (metadata, "nmm:episodeNumber", atoi (tag->value));
		}

		if ((tag = find_tag (format, video_stream, NULL, "season_number"))) {
			tracker_resource_set_int64 (metadata, "nmm:season", atoi (tag->value));
		}

		if ((tag = find_tag (format, video_stream, NULL, "creation_time"))) {
			content_created = tracker_date_guess (tag->value);
			if (content_created) {
				tracker_resource_set_string (metadata, "nie:contentCreated", content_created);
			}
		}

		if ((tag = find_tag (format, audio_stream, NULL, "performer"))) {
			g_autoptr (TrackerResource) performer = NULL;

			performer = tracker_extract_new_artist (tag->value);
			tracker_resource_set_relation (metadata, "nmm:leadActor", performer);
		}

		if ((tag = find_tag (format, audio_stream, NULL, "composer"))) {
			g_autoptr (TrackerResource) composer = NULL;

			composer = tracker_extract_new_artist (tag->value);
			tracker_resource_set_relation (metadata, "nmm:director", composer);
		}

		if ((hash = extract_gibest_hash (file))) {
			g_autofree char *hash_str;

			hash_str = g_strdup_printf ("%" G_GINT64_MODIFIER "x", hash);
			add_hash (metadata, file, hash_str, "gibest");
		}
	} else if (audio_stream) {
		g_autoptr (TrackerResource) album_artist = NULL, artist = NULL, performer = NULL;
		g_autofree char *album_artist_name = NULL;
		g_autofree char *album_title = NULL;
		TrackerToc *cue_sheet = NULL;
		int track_count = 0;

		tracker_resource_add_uri (metadata, "rdf:type", "nmm:MusicPiece");
		tracker_resource_add_uri (metadata, "rdf:type", "nfo:Audio");

		if (audio_stream->duration > 0) {
			gint64 duration = av_rescale(audio_stream->duration, audio_stream->time_base.num,
			                             audio_stream->time_base.den);
			tracker_resource_set_int64 (metadata, "nfo:duration", duration);
		}

		if ((tag = find_tag (format, audio_stream, NULL, "track"))) {
			int track = 0;

			if (sscanf (tag->value, "%u/%u", &track, &track_count) != 2) {
				track = atoi (tag->value);
			}

			if (track > 0)
				tracker_resource_set_int64 (metadata, "nmm:trackNumber", track);
		}

		if (track_count == 0 &&
		    (tag = find_tag (format, audio_stream, NULL, "tracktotal"))) {
			track_count = atoi (tag->value);
		}

		if ((tag = find_tag (format, audio_stream, NULL, "album"))) {
			album_title = g_strdup (tag->value);
		}

		if (album_title && (tag = find_tag (format, audio_stream, NULL, "album_artist"))) {
			album_artist_name = g_strdup (tag->value);
			album_artist = tracker_extract_new_artist (album_artist_name);
		}

		if ((tag = find_tag (format, audio_stream, NULL, "artist"))) {
			artist = tracker_extract_new_artist (tag->value);
			tracker_resource_set_relation (metadata, "nmm:artist", artist);

			if ((tag = find_tag (format, audio_stream, NULL, "musicbrainz_artistid"))) {
				add_external_reference (artist,
				                        "https://musicbrainz.org/artist",
				                        tag->value,
				                        "https://musicbrainz.org/doc/Artist");
			}
		}

		if ((tag = find_tag (format, audio_stream, NULL, "performer"))) {
			performer = tracker_extract_new_artist (tag->value);
			tracker_resource_set_relation (metadata, "nmm:performer", performer);
		}

		if ((tag = find_tag (format, audio_stream, NULL, "date"))) {
			content_created = tracker_date_guess (tag->value);
			if (content_created) {
				tracker_resource_set_string (metadata, "nie:contentCreated", content_created);
			}
		}

		if ((tag = find_tag (format, audio_stream, NULL, "acoustid_fingerprint"))) {
			add_hash (metadata, file, tag->value, "chromaprint");
		}

		if ((tag = find_tag (format, audio_stream, NULL, "musicbrainz_trackid"))) {
			add_external_reference (metadata,
			                        "https://musicbrainz.org/recording",
			                        tag->value,
			                        "https://musicbrainz.org/doc/Recording");
		}

		if ((tag = find_tag (format, audio_stream, NULL, "musicbrainz_releasetrackid"))) {
			add_external_reference (metadata,
			                        "https://musicbrainz.org/track",
			                        tag->value,
			                        "https://musicbrainz.org/doc/Track");
		}

		if ((tag = find_tag (format, audio_stream, NULL, "composer"))) {
			g_autoptr (TrackerResource) composer = tracker_extract_new_artist (tag->value);
			tracker_resource_set_relation (metadata, "nmm:composer", composer);
		}

		if (album_title) {
			int disc_number = 1;
			g_autoptr (TrackerResource) album_disc = NULL;
			TrackerResource *album;

			if ((tag = find_tag (format, audio_stream, NULL, "disc"))) {
				disc_number = atoi (tag->value);
			}

			album_disc = tracker_extract_new_music_album_disc (album_title, album_artist, disc_number, content_created);
			tracker_resource_set_relation (metadata, "nmm:musicAlbumDisc", album_disc);

			album = tracker_resource_get_first_relation (album_disc, "nmm:albumDiscAlbum");
			tracker_resource_set_relation (metadata, "nmm:musicAlbum", album);

			if (track_count > 0)
				tracker_resource_set_int (album, "nmm:albumTrackCount", track_count);

			if ((tag = find_tag (format, audio_stream, NULL, "musicbrainz_albumid"))) {
				add_external_reference (album,
				                        "https://musicbrainz.org/release",
				                        tag->value,
				                        "https://musicbrainz.org/doc/Release");
			}

			if ((tag = find_tag (format, audio_stream, NULL, "musicbrainz_releasegroupid"))) {
				add_external_reference (album,
				                        "https://musicbrainz.org/release-group",
				                        tag->value,
				                        "https://musicbrainz.org/doc/Release_Group");
			}
		}

		if ((tag = find_tag (format, audio_stream, NULL, "cuesheet"))) {
			cue_sheet = tracker_cue_sheet_parse (tag->value);
		} else {
			cue_sheet = tracker_cue_sheet_guess_from_uri (uri);
		}

		if (cue_sheet) {
			tracker_cue_sheet_apply_to_resource (cue_sheet,
			                                     metadata,
			                                     info);
			tracker_toc_free (cue_sheet);
		}
	}

	if ((tag = find_tag (format, audio_stream, video_stream, "xmp"))) {
		TrackerXmpData *xmp;

		xmp = tracker_xmp_new (tag->value, -1, uri);
		tracker_xmp_apply_to_resource (metadata, xmp);
		tracker_xmp_free (xmp);
	}

	if (format->bit_rate > 0) {
		tracker_resource_set_int64 (metadata, "nfo:averageBitrate", format->bit_rate);
	}

	if ((tag = find_tag (format, audio_stream, video_stream, "comment"))) {
		tracker_resource_set_string (metadata, "nie:comment", tag->value);
	}

	if ((tag = find_tag (format, audio_stream, video_stream, "copyright"))) {
		tracker_resource_set_string (metadata, "nie:copyright", tag->value);
	}

	if ((tag = find_tag (format, audio_stream, video_stream, "description"))) {
		tracker_resource_set_string (metadata, "nie:description", tag->value);
	}

	if ((tag = find_tag (format, audio_stream, video_stream, "genre"))) {
		tracker_resource_set_string (metadata, "nfo:genre", tag->value);
	}

	if ((tag = find_tag (format, audio_stream, video_stream, "encoder"))) {
		tracker_resource_set_string (metadata, "nie:generator", tag->value);
	}

	if ((tag = find_tag (format, audio_stream, video_stream, "title"))) {
		title = tag->value;
	}

	tracker_guarantee_resource_title_from_file (metadata, "nie:title", title, uri, NULL);

#ifdef HAVE_GUPNP_DLNA
	gupnp_info = tracker_gupnp_dlna_information_new (format,
	                                                 audio_stream,
	                                                 video_stream);
	profile_guesser = gupnp_dlna_profile_guesser_new (FALSE, FALSE);

	gupnp_profile = gupnp_dlna_profile_guesser_guess_profile_from_info (profile_guesser,
	                                                                    gupnp_info);

	if (gupnp_profile) {
		const char *profile_name, *profile_mime;

		profile_mime = gupnp_dlna_profile_get_mime (gupnp_profile);
		if (profile_mime)
			tracker_resource_set_string (metadata, "nmm:dlnaMime", profile_mime);

		profile_name = gupnp_dlna_profile_get_name (gupnp_profile);
		if (profile_name)
			tracker_resource_set_string (metadata, "nmm:dlnaProfile", profile_name);
	}
#endif

	avformat_close_input (&format);

	tracker_extract_info_set_resource (info, metadata);

	return TRUE;
}

G_MODULE_EXPORT gboolean
tracker_extract_module_init (GError **error)
{
	av_log_set_level (AV_LOG_FATAL);

	return TRUE;
}
