/*
 * Copyright (C) 2011, ARQ Media <sam.thursfield@codethink.co.uk>
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
 *
 * Author: Sam Thursfield <sam.thursfield@codethink.co.uk>
 */

#include "config-miners.h"

#include <stdlib.h>
#include <string.h>

#include <glib.h>
#include <gio/gio.h>

#if defined(HAVE_LIBCUE)
#include <libcue.h>
#endif

#include <tracker-common.h>

#include "utils/tracker-extract.h"

#include "tracker-cue-sheet.h"
#include "tracker-main.h"

#if defined(HAVE_LIBCUE)

struct _TrackerToc {
	Cd *cue_data;
};

static TrackerToc *
tracker_toc_new (void)
{
	TrackerToc *toc;

	toc = g_new0 (TrackerToc, 1);

	return toc;
}

void
tracker_toc_free (TrackerToc *toc)
{
	if (!toc) {
		return;
	}

	cd_delete (toc->cue_data);
	g_free (toc);
}

static void
set_cdtext_resource_string (Cdtext          *cd_text,
                            enum Pti         index,
                            TrackerResource *resource,
                            const gchar     *property)
{
	const gchar *text;

	text = cdtext_get (index, cd_text);
	if (text)
		tracker_resource_set_string (resource, property, text);
}

static void
set_rem_resource_double (Rem             *cd_comments,
                         enum RemType     index,
                         TrackerResource *resource,
                         const gchar     *property)
{
	const gchar *text;
	gdouble value;

	text = rem_get (index, cd_comments);
	if (!text)
		return;

	value = strtod (text, NULL);
	if (value != 0.0)
		tracker_resource_set_double (resource, property, value);
}

/* This function runs in two modes: for external CUE sheets, it will check
 * the FILE field for each track and build a TrackerToc for all the tracks
 * contained in @file_name. If @file_name does not appear in the CUE sheet,
 * %NULL will be returned. For embedded CUE sheets, @file_name will be NULL
 * the whole TOC will be returned regardless of any FILE information.
 */
static TrackerToc *
parse_cue_sheet_for_file (const gchar *cue_sheet,
                          const gchar *file_name)
{
	TrackerToc *toc = NULL;
	Cd *cd;
	Track *track;
	gint i;

	cd = cue_parse_string (cue_sheet);

	if (cd == NULL) {
		g_debug ("Unable to parse CUE sheet for %s.",
		         file_name ? file_name : "(embedded in FLAC)");
		return NULL;
	}

	for (i = 1; i <= cd_get_ntrack (cd); i++) {
		track = cd_get_track (cd, i);

		/* CUE sheets generally have the correct basename but wrong
		 * extension in the FILE field, so this is what we test for.
		 */
		if (file_name != NULL) {
			if (!tracker_filename_casecmp_without_extension (file_name,
			                                                 track_get_filename (track))) {
				continue;
			}
		}

		if (track_get_mode (track) != MODE_AUDIO)
			continue;

		toc = tracker_toc_new ();
		toc->cue_data = cd;
		break;
	}

	if (!toc)
		cd_delete (cd);

	return toc;
}

TrackerToc *
tracker_cue_sheet_parse (const gchar *cue_sheet)
{
	TrackerToc *result;

	result = parse_cue_sheet_for_file (cue_sheet, NULL);

	return result;
}

static GList *
find_local_cue_sheets (GFile *audio_file)
{
	TrackerSparqlConnection *conn;
	g_autoptr (TrackerSparqlStatement) stmt = NULL;
	g_autoptr (TrackerSparqlCursor) cursor = NULL;
	g_autoptr (GFile) parent = NULL;
	g_autofree gchar *parent_uri = NULL;
	GList *result = NULL;

	conn = tracker_main_get_connection ();
	stmt = tracker_sparql_connection_load_statement_from_gresource (conn,
	                                                                "/org/freedesktop/Tracker3/Extract/queries/get-cue-sheets.rq",
	                                                                NULL, NULL);
	if (!stmt)
		return NULL;

	parent = g_file_get_parent (audio_file);
	parent_uri = g_file_get_uri (parent);
	tracker_sparql_statement_bind_string (stmt, "parent", parent_uri);
	cursor = tracker_sparql_statement_execute (stmt, NULL, NULL);

	if (!cursor)
		return NULL;

	while (tracker_sparql_cursor_next (cursor, NULL, NULL)) {
		const gchar *str;

		str = tracker_sparql_cursor_get_string (cursor, 0, NULL);
		result = g_list_prepend (result, g_file_new_for_uri (str));
	}

	return result;
}

static GFile *
find_matching_cue_file (GFile *audio_file)
{
	const gchar *dot;
	g_autofree gchar *uri = NULL, *cue_uri = NULL;
	g_autoptr (GFile) file = NULL;

	uri = g_file_get_uri (audio_file);
	dot = strrchr (uri, '.');
	if (!dot)
		return NULL;

	cue_uri = g_strdup_printf ("%.*s.cue", (int) (dot - uri), uri);
	file = g_file_new_for_uri (cue_uri);

	if (g_file_query_exists (file, NULL))
		return g_steal_pointer (&file);

	return NULL;
}

TrackerToc *
tracker_cue_sheet_guess_from_uri (const gchar *uri)
{
	GFile *audio_file;
	GFile *cue_sheet_file;
	gchar *audio_file_name;
	GList *cue_sheet_list = NULL;
	TrackerToc *toc;
	GError *error = NULL;
	GList *n;

	audio_file = g_file_new_for_uri (uri);
	audio_file_name = g_file_get_basename (audio_file);

	cue_sheet_file = find_matching_cue_file (audio_file);

	if (cue_sheet_file)
		cue_sheet_list = g_list_prepend (cue_sheet_list, cue_sheet_file);
	else
		cue_sheet_list = find_local_cue_sheets (audio_file);

	toc = NULL;

	for (n = cue_sheet_list; n != NULL; n = n->next) {
		gchar *buffer;

		cue_sheet_file = n->data;

		if (!g_file_load_contents (cue_sheet_file, NULL, &buffer, NULL, NULL, &error)) {
			g_debug ("Unable to read cue sheet: %s", error->message);
			g_error_free (error);
			continue;
		}

		toc = parse_cue_sheet_for_file (buffer, audio_file_name);

		g_free (buffer);

		if (toc != NULL) {
			char *path = g_file_get_path (cue_sheet_file);
			g_debug ("Using external CUE sheet: %s", path);
			g_free (path);
			break;
		}
	}

	g_list_foreach (cue_sheet_list, (GFunc) g_object_unref, NULL);
	g_list_free (cue_sheet_list);

	g_object_unref (audio_file);
	g_free (audio_file_name);

	return toc;
}

static TrackerResource *
intern_artist (GHashTable *artists,
               const char *name)
{
	TrackerResource *resource;

	resource = g_hash_table_lookup (artists, name);
	if (resource)
		return g_object_ref (resource);

	resource = tracker_extract_new_artist (name);
	g_hash_table_insert (artists,
	                     g_strdup (name),
	                     g_object_ref (resource));

	return resource;
}

static TrackerResource *
new_album_from_cue_sheet (TrackerToc *toc,
                          GHashTable *artists)
{
	g_autoptr (TrackerResource) album_artist = NULL;
	const char *album_title = NULL;
	g_autofree char *date = NULL;
	Cdtext *cd_text;
	Rem *remarks;

	cd_text = cd_get_cdtext (toc->cue_data);
	remarks = cd_get_rem (toc->cue_data);

	if (cd_text) {
		const char *text;

		album_title = cdtext_get (PTI_TITLE, cd_text);
		text = cdtext_get (PTI_PERFORMER, cd_text);
		if (text)
			album_artist = intern_artist (artists, text);
	}

	if (!album_title)
		return NULL;

	if (remarks) {
		const char *text;
		int year;

		g_autoptr (GDateTime) datetime = NULL;

		text = rem_get (REM_DATE, remarks);
		if (text) {
			g_autoptr (GTimeZone) tz = NULL;

			tz = g_time_zone_new_utc ();
			year = atoi (text);
			datetime = g_date_time_new (tz, year, 1, 1, 0, 0, 0);
			date = g_date_time_format_iso8601 (datetime);
		}
	}

	return tracker_extract_new_music_album_disc (album_title,
	                                             album_artist,
	                                             1, date);
}

static void
copy_property (TrackerResource *resource,
               TrackerResource *source,
               const gchar     *property)
{
	g_autoptr (GList) values = NULL;
	GList *l;

	values = tracker_resource_get_values (source, property);
	for (l = values; l; l = l->next)
		tracker_resource_add_gvalue (resource, property, l->data);
}

void
tracker_cue_sheet_apply_to_resource (TrackerToc         *toc,
                                     TrackerResource    *ie,
                                     TrackerExtractInfo *info)
{
	TrackerResource *ie_performer, *ie_composer;
	g_autoptr (TrackerResource) file_resource = NULL, album_disc = NULL, album = NULL;
	g_autofree char *basename = NULL, *uri = NULL;
	g_autoptr (GHashTable) artists = NULL;
	gint64 total_duration = 0;
	GFile *file;
	int i;

	file = tracker_extract_info_get_file (info);
	uri = g_file_get_uri (file);
	basename = g_file_get_basename (file);

	artists = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_object_unref);

	g_set_object (&album_disc, tracker_resource_get_first_relation (ie, "nmm:musicAlbumDisc"));
	if (!album_disc)
		album_disc = new_album_from_cue_sheet (toc, artists);

	if (album_disc)
		g_set_object (&album, tracker_resource_get_first_relation (album_disc, "nmm:albumDiscAlbum"));

	if (album) {
		Rem *cd_remarks;

		cd_remarks = cd_get_rem (toc->cue_data);
		if (cd_remarks) {
			set_rem_resource_double (cd_remarks,
			                         REM_REPLAYGAIN_ALBUM_GAIN,
			                         album, "nfo:albumGain");
			set_rem_resource_double (cd_remarks,
			                         REM_REPLAYGAIN_ALBUM_PEAK,
			                         album, "nmm:albumPeakGain");
		}

		tracker_resource_set_int (album, "nmm:albumTrackCount", cd_get_ntrack (toc->cue_data));
	}

	/* Extract existing information from the given resource */
	ie_performer = tracker_resource_get_first_relation (ie, "nmm:performer");
	ie_composer = tracker_resource_get_first_relation (ie, "nmm:composer");
	total_duration = tracker_resource_get_first_int64 (ie, "nfo:duration");

	g_set_object (&file_resource, tracker_resource_get_first_relation (ie, "nie:isStoredAs"));
	if (!file_resource) {
		file_resource = tracker_resource_new (uri);
		tracker_resource_add_uri (file_resource, "rdf:type", "nie:DataObject");
		tracker_resource_set_relation (ie, "nie:isStoredAs", file_resource);
	}

	for (i = 1; i <= cd_get_ntrack (toc->cue_data); i++) {
		g_autoptr (TrackerResource) track_resource = NULL, performer = NULL, composer = NULL;
		Track *track;
		Cdtext *cd_text;
		Rem *cd_remarks;
		gint64 duration;
		gdouble start;

		track = cd_get_track (toc->cue_data, i);

		/* CUE sheets generally have the correct basename but wrong
		 * extension in the FILE field, so this is what we test for.
		 */
		if (!tracker_filename_casecmp_without_extension (basename,
		                                                 track_get_filename (track))) {
			continue;
		}

		if (track_get_mode (track) != MODE_AUDIO)
			continue;

		/* Reuse the "root" InformationElement resource for the first track,
		 * so there's no spare ones.
		 */
		if (i == 1) {
			g_set_object (&track_resource, ie);
		} else {
			g_autofree char *suffix = NULL, *resource_uri = NULL;

			suffix = g_strdup_printf ("%d", i);
			resource_uri = tracker_extract_info_get_content_id (info, suffix);

			track_resource = tracker_resource_new (resource_uri);
			tracker_resource_add_uri (track_resource, "rdf:type", "nmm:MusicPiece");
			tracker_resource_add_uri (track_resource, "rdf:type", "nfo:Audio");
			tracker_resource_set_uri (track_resource, "nie:isStoredAs", uri);

			copy_property (track_resource, ie, "nfo:channels");
			copy_property (track_resource, ie, "nfo:averageBitrate");
			copy_property (track_resource, ie, "nfo:sampleRate");
			copy_property (track_resource, ie, "nie:generator");

			tracker_resource_add_relation (file_resource, "nie:interpretedAs", track_resource);
		}

		duration = (gint64) track_get_length (track) / 75;
		start = track_get_start (track) / 75.0;

		if (duration > 0) {
			tracker_resource_set_int64 (track_resource, "nfo:duration", duration);
		} else if (i == cd_get_ntrack (toc->cue_data) && total_duration > start) {
			/* The last element may not have a duration, because it depends
			 * on the duration of the media file rather than info from the
			 * cue sheet. In this case figure the data out from the total
			 * duration.
			 */
			tracker_resource_set_int64 (track_resource, "nfo:duration", total_duration - start);
		}

		tracker_resource_set_double (track_resource, "nfo:audioOffset", start);

		cd_text = track_get_cdtext (track);
		if (cd_text) {
			const char *text;

			text = cdtext_get (PTI_PERFORMER, cd_text);
			if (text)
				performer = intern_artist (artists, text);

			text = cdtext_get (PTI_COMPOSER, cd_text);
			if (text)
				composer = intern_artist (artists, text);

			set_cdtext_resource_string (cd_text, PTI_TITLE,
			                            track_resource, "nie:title");
		}

		/* Set from embedded metadata if empty in cue sheet */
		if (!performer)
			g_set_object (&performer, ie_performer);
		if (!composer)
			g_set_object (&composer, ie_composer);

		cd_remarks = track_get_rem (track);
		if (cd_remarks) {
			set_rem_resource_double (cd_remarks,
			                         REM_REPLAYGAIN_TRACK_GAIN,
			                         track_resource, "nfo:gain");
			set_rem_resource_double (cd_remarks,
			                         REM_REPLAYGAIN_TRACK_PEAK,
			                         track_resource, "nfo:peakGain");
		}

		tracker_resource_set_int (track_resource, "nmm:trackNumber", i);

		if (album)
			tracker_resource_set_relation (track_resource, "nmm:musicAlbum", album);
		if (album_disc)
			tracker_resource_set_relation (track_resource, "nmm:musicAlbumDisc", album_disc);
		if (performer)
			tracker_resource_set_relation (track_resource, "nmm:performer", performer);
		if (composer)
			tracker_resource_set_relation (track_resource, "nmm:composer", composer);
	}
}

#else  /* ! HAVE_LIBCUE */

TrackerToc *
tracker_cue_sheet_parse (const gchar *cue_sheet)
{
	return NULL;
}

TrackerToc *
tracker_cue_sheet_guess_from_uri (const gchar *uri)
{
	return NULL;
}

void
tracker_cue_sheet_apply_to_resource (TrackerToc         *toc,
                                     TrackerResource    *ie,
                                     TrackerExtractInfo *info)
{
}

void
tracker_toc_free (TrackerToc *toc)
{
}

#endif /* ! HAVE_LIBCUE */
