/*
 * repository.c
 * This file is part of libgit2-glib
 *
 * Copyright (C) 2011 - Paolo Borelli
 *
 * libgit2-glib is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libgit2-glib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libgit2-glib. If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#include <glib.h>
#include <glib/gstdio.h>

#include "libgit2-glib/ggit.h"

#define TESTREPO_NAME "testrepo.git"

typedef struct {
	gchar *git_dir;
} TestFixture;

/*
 * This function is copied from
 * http://bugzilla.gnome.org/show_bug.cgi?id=524831
 * and renamed from g_mkdtemp to _g_mkdtemp.
 *
 * If/when this function gets added to glib, it can be removed.
 *
 *
 * g_mkdtemp:
 * @tmpl: template directory name
 *
 * Creates a temporary directory. See the mkdtemp() documentation
 * on most UNIX-like systems.
 *
 * The parameter is a string that should follow the rules for
 * mkdtemp() templates, i.e. contain the string "XXXXXX".  g_mkdtemp()
 * is slightly more flexible than mkdtemp() in that the sequence does
 * not have to occur at the very end of the template. The X string
 * will be modified to form the name of a directory that didn't
 * already exist.  The string should be in the GLib file name
 * encoding. Most importantly, on Windows it should be in UTF-8.
 *
 * Return value: If a temporary directory was successfully created,
 * @tmpl will be returned with the XXXXXX string modified in such a
 * way as to make the path unique.  In case of errors, %NULL is
 * returned.
 */
static gchar *
_g_mkdtemp (gchar *tmpl)
{
	static const char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	static const int NLETTERS = sizeof (letters) - 1;
	static int counter = 0;
	char *XXXXXX;
	gint64 time;
	glong value;
	int count;

	/* find the last occurrence of "XXXXXX" */
	XXXXXX = g_strrstr (tmpl, "XXXXXX");

	if (!XXXXXX || strncmp (XXXXXX, "XXXXXX", 6))
	{
		errno = EINVAL;
		return NULL;
	}

	/* Get some more or less random data.  */
	time = g_get_real_time ();
	value = ((time % G_USEC_PER_SEC) ^ (time / G_USEC_PER_SEC)) + counter++;

	for (count = 0; count < 100; value += 7777, ++count)
	{
		glong v = value;

		/* Fill in the random bits.  */
		XXXXXX[0] = letters[v % NLETTERS];
		v /= NLETTERS;
		XXXXXX[1] = letters[v % NLETTERS];
		v /= NLETTERS;
		XXXXXX[2] = letters[v % NLETTERS];
		v /= NLETTERS;
		XXXXXX[3] = letters[v % NLETTERS];
		v /= NLETTERS;
		XXXXXX[4] = letters[v % NLETTERS];
		v /= NLETTERS;
		XXXXXX[5] = letters[v % NLETTERS];

		/* tmpl is in UTF-8 on Windows, thus use g_mkdir() */
		if (g_mkdir (tmpl, 0700) == 0)
			return tmpl;

		if (errno != EEXIST)
			 /* Any other error will apply also to other names we might
			 *  try, and there are 2^32 or so of them, so give up now.
			 */
			 return NULL;
	}

	/* We got out of the loop because we ran out of combinations to try.  */
	errno = EEXIST;
	return NULL;
}

static void
test_setup (TestFixture   *fixture,
            gconstpointer  data)
{
	fixture->git_dir = g_build_filename (g_get_tmp_dir (),
	                                     TESTREPO_NAME "XXXXXX",
	                                     NULL);

	fixture->git_dir = _g_mkdtemp (fixture->git_dir);
}

static void
test_teardown (TestFixture   *fixture,
               gconstpointer  data)
{
	gchar *cmd;
	gint ret;

	/* remove all files in the directory, it sucks that
	 * there isn't a glib util for this, for now lets just cheat */
	cmd = g_strdup_printf ("rm -rf '%s'", fixture->git_dir);
	ret = system (cmd);
	g_assert (ret == 0);

	g_free (cmd);
	g_free (fixture->git_dir);
}

static void
test_runner (TestFixture   *fixture,
             gconstpointer  data)
{
	((void (*) (const gchar *git_dir)) data) (fixture->git_dir);
}

static void
do_test_init (const gchar *git_dir,
              gboolean     bare)
{
	GError *err = NULL;
	GgitRepository *repo;
	GFile *f;
	GFile *dotgit;

	f = g_file_new_for_path (git_dir);
	repo = ggit_repository_init_repository (f, bare, &err);

	g_assert_no_error (err);

	dotgit = bare ? g_file_dup (f) : g_file_get_child (f, ".git");
	g_object_unref (f);

	f = ggit_repository_get_location (repo);

	g_assert (g_file_equal (dotgit, f));
	g_assert (g_file_query_exists (dotgit, NULL));

	g_object_unref (dotgit);
	g_object_unref (f);

	g_assert (ggit_repository_is_bare (repo) == bare);

	g_object_unref (repo);
}

static void
test_repository_init (const gchar *git_dir)
{
	do_test_init (git_dir, FALSE);
}

static void
test_repository_init_bare (const gchar *git_dir)
{
	do_test_init (git_dir, TRUE);
}

static void
test_repository_blob_stream (const gchar *git_dir)
{
	GFile *f;
	GgitRepository *repo;
	GError *err = NULL;
	GgitBlobOutputStream *stream;
	GgitOId *oid;
	const gchar *msg = "hello world\n";
	gsize written;
	gsize msglen;
	GgitBlob *blob;
	gsize rl;
	const guchar *content;

	f = g_file_new_for_path (git_dir);
	repo = ggit_repository_init_repository (f, FALSE, &err);
	g_object_unref (f);

	g_assert_no_error (err);

	stream = ggit_repository_create_blob (repo);

	msglen = strlen (msg);

	written = g_output_stream_write (G_OUTPUT_STREAM (stream),
	                                 msg,
	                                 msglen,
	                                 NULL,
	                                 &err);

	g_assert_no_error (err);
	g_assert_cmpint (written, ==, msglen);

	g_output_stream_close (G_OUTPUT_STREAM (stream), NULL, &err);

	g_assert_no_error (err);
	oid = ggit_blob_output_stream_get_id (stream, &err);

	g_assert_no_error (err);
	g_assert (oid);

	blob = (GgitBlob *)ggit_repository_lookup (repo,
	                                           oid,
	                                           GGIT_TYPE_BLOB,
	                                           &err);

	g_assert_no_error (err);
	g_assert (blob);

	content = ggit_blob_get_raw_content (blob, &rl);
	g_assert_cmpint (rl, ==, msglen);
	g_assert_cmpint (memcmp (content, msg, msglen), ==, 0);

	ggit_oid_free (oid);

	g_object_unref (stream);
	g_object_unref (repo);
}

static void
test_repository_encoding (const gchar *git_dir)
{
	GFile *f;
	GError *err = NULL;
	GgitOId *cid;
	GgitTree *tree;
	GgitIndex *idx;
	GFile *afile;
	GgitOId *toid;
	GgitCommit *commit;
	GgitRepository *repo;
	GgitSignature *author;
	const gchar *msg;

	const char *encoding = "IBM862";
	const char *message = "\x80\x89\x8f\x20\x84\x90\x87\x9a\x85\x8d\x20\x8e\x92\x89\x83\x20\x92\x8c\x20\x92\x89\x91\x9a\x85";
	const char *message_utf8 = "\xd7\x90\xd7\x99\xd7\x9f\x20\xd7\x94\xd7\xa0\xd7\x97\xd7\xaa\xd7\x95\xd7\x9d\x20\xd7\x9e\xd7\xa2\xd7\x99\xd7\x93\x20\xd7\xa2\xd7\x9c\x20\xd7\xa2\xd7\x99\xd7\xa1\xd7\xaa\xd7\x95";

	f = g_file_new_for_path (git_dir);
	repo = ggit_repository_init_repository (f, FALSE, &err);

	g_assert_no_error (err);
	g_assert (repo != NULL);

	idx = ggit_repository_get_index (repo, &err);

	g_assert_no_error (err);
	g_assert (idx != NULL);

	afile = g_file_get_child (f, "a");
	g_object_unref (f);

	g_file_replace_contents (afile,
	                         message,
	                         strlen (message),
	                         NULL,
	                         FALSE,
	                         G_FILE_CREATE_NONE,
	                         NULL,
	                         NULL,
	                         &err);
	g_assert_no_error (err);

	ggit_index_add_file (idx, afile, &err);
	g_assert_no_error (err);
	g_object_unref (afile);

	toid = ggit_index_write_tree (idx, &err);
	g_assert_no_error (err);
	g_assert (toid != NULL);

	tree = GGIT_TREE (ggit_repository_lookup (repo, toid, GGIT_TYPE_TREE, &err));
	g_assert_no_error (err);
	g_assert (tree != NULL);

	ggit_oid_free (toid);
	g_object_unref (idx);

	author = ggit_signature_new_now ("Jesse van den Kieboom",
	                                 "jessevdk@gnome.org",
	                                 &err);

	g_assert_no_error (err);
	g_assert (author != NULL);

	cid = ggit_repository_create_commit (repo,
	                                     "HEAD",
	                                     author,
	                                     author,
	                                     encoding,
	                                     message,
	                                     tree,
	                                     NULL,
	                                     0,
	                                     &err);

	g_assert_no_error (err);
	g_assert (cid != NULL);

	g_object_unref (author);
	g_object_unref (tree);

	commit = GGIT_COMMIT (ggit_repository_lookup (repo, cid, GGIT_TYPE_COMMIT, &err));
	g_assert_no_error (err);
	g_assert (commit != NULL);

	ggit_oid_free (cid);

	g_assert_cmpstr (ggit_commit_get_message_encoding (commit),
	                 ==,
	                 encoding);

	msg = ggit_commit_get_message (commit);
	g_assert_cmpstr (msg, ==, message_utf8);

	g_object_unref (commit);
	g_object_unref (repo);
}

int
main (int    argc,
      char **argv)
{
	ggit_init ();

	g_test_init (&argc, &argv, NULL);

#define TEST(path, ftest) \
  g_test_add ("/repository/" path, TestFixture, \
              (gpointer) test_repository_##ftest, \
              test_setup, test_runner, test_teardown)

	TEST ("init", init);
	TEST ("init-bare", init_bare);
	TEST ("blob-stream", blob_stream);
	TEST ("encoding", encoding);

	return g_test_run ();
}

/* ex:set ts=8 noet: */
