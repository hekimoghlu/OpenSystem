/*
 * gnome-keyring
 *
 * Copyright (C) 2008 Stefan Walter
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, see
 * <http://www.gnu.org/licenses/>.
 */

#include "config.h"

#include "gkm-marshal.h"
#include "gkm-transaction.h"

#include <glib/gstdio.h>

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#ifndef O_BINARY
# define O_BINARY 0
#endif

enum {
	PROP_0,
	PROP_COMPLETED,
	PROP_FAILED,
	PROP_RESULT
};

enum {
	COMPLETE,
	LAST_SIGNAL
};

static guint signals[LAST_SIGNAL] = { 0 };

struct _GkmTransaction {
	GObject parent;
	GList *completes;
	gboolean failed;
	gboolean completed;
	CK_RV result;
};

typedef struct _Complete {
	GObject *object;
	GkmTransactionFunc func;
	gpointer user_data;
} Complete;

G_DEFINE_TYPE (GkmTransaction, gkm_transaction, G_TYPE_OBJECT);

#define MAX_TRIES 100000

/* -----------------------------------------------------------------------------
 * INTERNAL
 */

static gboolean
complete_invoke (GkmTransaction *transaction, Complete *complete)
{
	g_assert (complete);
	g_assert (complete->func);

	return (complete->func) (transaction, complete->object, complete->user_data);
}

static void
complete_destroy (Complete *complete)
{
	g_assert (complete->func);
	if (complete->object)
		g_object_unref (complete->object);
	g_slice_free (Complete, complete);
}

static gboolean
complete_accumulator (GSignalInvocationHint *ihint, GValue *return_accu,
                      const GValue *handler_return, gpointer data)
{
	gboolean result;

	/* If any of them return false, then the result is false */
	result = g_value_get_boolean (handler_return);
	if (result == FALSE)
		g_value_set_boolean (return_accu, FALSE);

	/* Continue signal invocations */
	return TRUE;
}

static gboolean
complete_new_file (GkmTransaction *self, GObject *unused, gpointer user_data)
{
	gchar *path = user_data;
	gboolean ret = TRUE;

	if (gkm_transaction_get_failed (self)) {
		if (g_unlink (path) < 0) {
			g_warning ("couldn't delete aborted file, data may be lost: %s: %s",
			           path, g_strerror (errno));
			ret = FALSE;
		}
	}

	g_free (path);
	return ret;
}

static gboolean
begin_new_file (GkmTransaction *self, const gchar *filename)
{
	g_assert (GKM_IS_TRANSACTION (self));
	g_assert (!gkm_transaction_get_failed (self));
	g_assert (filename);

	gkm_transaction_add (self, NULL, complete_new_file, g_strdup (filename));
	return TRUE;
}

static gboolean
complete_link_temporary (GkmTransaction *self, GObject *unused, gpointer user_data)
{
	gchar *path = user_data;
	gboolean ret = TRUE;
	gchar *original;
	gchar *ext;

	/* When failed, rename temporary back */
	if (gkm_transaction_get_failed (self)) {

		/* Figure out the original file name */
		original = g_strdup (path);
		ext = strrchr (original, '.');
		g_return_val_if_fail (ext, FALSE);
		*ext = '\0';

		/* Now rename us back */
		if (g_rename (path, original) == -1) {
			g_warning ("couldn't restore original file, data may be lost: %s: %s",
			           original, g_strerror (errno));
			ret = FALSE;
		}

		g_free (original);

	/* When succeeded, remove temporary */
	} else {
		if (g_unlink (path) == -1) {
			g_warning ("couldn't delete temporary backup file: %s: %s",
			           path, g_strerror (errno));
			ret = TRUE; /* Not actually that bad of a situation */
		}
	}

	g_free (path);
	return ret;
}


/* Copy the file SRCNAME to the file DSTNAME.  If DSTNAME already
   exists -1 is returned and ERRNO set to EEXIST.  Returns 0 on
   success. */
static int
copy_to_temp_file (const char *dstname, const char *srcname)
{
	int dstfd, srcfd;
	int nread, nwritten;
	int saveerr;
	char *bufp;
	char buffer[512]; /* If you change this size, please also adjust */
	                  /* test-transaction.c:test_write_large_file.   */

	do {
		srcfd = g_open (srcname, (O_RDONLY | O_BINARY));
	} while (srcfd == -1 && errno == EINTR);
	if (srcfd == -1) {
		saveerr = errno;
		g_warning ("couldn't open file to make temporary copy from: %s: %s",
		           srcname, g_strerror (saveerr));
		errno = saveerr;
		return -1;
	}

	do {
		dstfd = g_open (dstname,
		                (O_WRONLY | O_CREAT | O_EXCL | O_BINARY),
		                (S_IRUSR | S_IWUSR));
	} while (dstfd == -1 && errno == EINTR);
	if (dstfd == -1) {
		saveerr = errno;
		close (srcfd);
		errno = saveerr;
		return -1;
	}

	while ((nread = read (srcfd, buffer, sizeof buffer))) {
		if (nread == -1 && errno == EINTR)
			continue;
		if (nread == -1) {
			saveerr = errno;
			g_warning ("error reading file to make temporary copy from: %s: %s",
			           srcname, g_strerror (saveerr));
			goto failure;
		}

		bufp = buffer;
		do {
			do {
				nwritten = write (dstfd, bufp, nread);
			} while (nwritten == -1 && errno == EINTR);
			if (nwritten == -1) {
				saveerr = errno;
				g_warning ("error wrinting to temporary file: %s: %s",
				           dstname, g_strerror (saveerr));
				goto failure;
			}
			g_return_val_if_fail (nwritten <= nread, -1);
			nread -= nwritten;
			bufp += nwritten;
		} while (nread > 0);
	}
	/* EOF reached.  */
	if (close (dstfd)) {
		saveerr = errno;
		g_warning ("error closing temporary file: %s: %s",
		           dstname, g_strerror (saveerr));
		goto failure;
	}
	close (srcfd);
	return 0;

failure:
	close (dstfd);  /* (Doesn't harm if we try a second time.) */
	if (g_unlink (dstname))
		g_warning ("couldn't remove temporary file: %s: %s",
		           dstname, g_strerror (saveerr));
	close (srcfd);
	errno = saveerr;
	return -1;
}


static gboolean
begin_link_temporary_if_exists (GkmTransaction *self, const gchar *filename, gboolean *exists)
{
	guint i = 0;

	g_assert (GKM_IS_TRANSACTION (self));
	g_assert (!gkm_transaction_get_failed (self));
	g_assert (filename);
	g_assert (exists);

	for (i = 0; i < MAX_TRIES; ++i) {
		struct stat sb;
		unsigned int nlink;
		int stat_failed = 0;

		*exists = TRUE;

		/* Try to link to random temporary file names.  We try
		 * to use a hardlink to create a copy but if that
		 * fails (i.e. not supported by the FS), we copy the
		 * entire file.  The result should be the same except
		 * that the file times will change if we need to
		 * rollback the transaction. */
		if (stat (filename, &sb)) {
			stat_failed = 1;
		} else {
			gchar *result;

			result = g_strdup_printf ("%s.temp-%d", filename,
			                          g_random_int_range (0, G_MAXINT));
			nlink = (unsigned int)sb.st_nlink;
			/* The result code of link(2) is not reliable.
			 * Unless it fails with EEXIST we stat the
			 * file to check for success.  Note that there
			 * is a race here: If another process adds a
			 * link to the source file between link and
			 * stat, the check on the increased link count
			 * will fail.  Fortunately the case for
			 * hardlinks are not working solves it.  */
			if (link (filename, result) && errno == EEXIST) {
				/* This is probably a valid error.
				 * Let us try another temporary file.  */
			} else if (stat (filename, &sb)) {
				stat_failed = 1;
			} else {
				if ((sb.st_nlink == nlink + 1)
				    || !copy_to_temp_file (result, filename)) {
					/* Either the link worked or
					 * the copy succeeded.  */
					gkm_transaction_add (self, NULL,
					                     complete_link_temporary,
					                     result);
					return TRUE;
				}
			}

			g_free (result);
		}

		if (stat_failed && (errno == ENOENT || errno == ENOTDIR)) {
			/* The original file does not exist */
			*exists = FALSE;
			return TRUE;
		}

		/* If exists, try again, otherwise fail */
		if (errno != EEXIST) {
			g_warning ("couldn't create temporary file for: %s: %s",
			           filename, g_strerror (errno));
			gkm_transaction_fail (self, CKR_DEVICE_ERROR);
			return FALSE;
		}
	}

	g_assert_not_reached ();
}

static gboolean
write_sync_close (int fd, const guchar *data, gsize n_data)
{
	int res;

	if (fd == -1)
		return FALSE;

	while (n_data > 0) {
		res = write (fd, data, n_data);
		if (res < 0) {
			if (errno != EINTR && errno != EAGAIN) {
				close (fd);
				return FALSE;
			}
			continue;
		}
		g_assert (res <= n_data);

		data += res;
		n_data -= res;
	}

#ifdef HAVE_FSYNC
	if (fsync (fd) < 0) {
		close (fd);
		return FALSE;
	}
#endif

	if (close (fd) < 0)
		return FALSE;

	return TRUE;
}

static gboolean
write_to_file (const gchar *filename, const guchar *data, gsize n_data)
{
	gchar *dirname;
	gchar *template;
	gboolean result;

	g_assert (filename);

	dirname = g_path_get_dirname (filename);
	template = g_build_filename (dirname, ".temp-XXXXXX", NULL);
	g_free (dirname);

	if (write_sync_close (g_mkstemp (template), data, n_data)) {
		result = g_rename (template, filename) == 0;
	} else {
		g_unlink (template);
		result = FALSE;
	}

	g_free (template);
	return result;
}

/* -----------------------------------------------------------------------------
 * OBJECT
 */

static gboolean
gkm_transaction_real_complete (GkmTransaction *self)
{
	GList *l;

	g_return_val_if_fail (!self->completed, FALSE);
	self->completed = TRUE;
	g_object_notify (G_OBJECT (self), "completed");

	for (l = self->completes; l; l = g_list_next (l)) {
		complete_invoke (self, l->data);
		complete_destroy (l->data);
	}

	g_list_free (self->completes);
	self->completes = NULL;

	return TRUE;
}

static void
gkm_transaction_init (GkmTransaction *self)
{

}

static void
gkm_transaction_dispose (GObject *obj)
{
	GkmTransaction *self = GKM_TRANSACTION (obj);

	if (!self->completed)
		gkm_transaction_complete (self);

	G_OBJECT_CLASS (gkm_transaction_parent_class)->dispose (obj);
}

static void
gkm_transaction_finalize (GObject *obj)
{
	GkmTransaction *self = GKM_TRANSACTION (obj);

	g_assert (!self->completes);
	g_assert (self->completed);

	G_OBJECT_CLASS (gkm_transaction_parent_class)->finalize (obj);
}

static void
gkm_transaction_set_property (GObject *obj, guint prop_id, const GValue *value,
                              GParamSpec *pspec)
{
	switch (prop_id) {
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, prop_id, pspec);
		break;
	}
}

static void
gkm_transaction_get_property (GObject *obj, guint prop_id, GValue *value,
                              GParamSpec *pspec)
{
	GkmTransaction *self = GKM_TRANSACTION (obj);

	switch (prop_id) {
	case PROP_COMPLETED:
		g_value_set_boolean (value, gkm_transaction_get_completed (self));
		break;
	case PROP_FAILED:
		g_value_set_boolean (value, gkm_transaction_get_failed (self));
		break;
	case PROP_RESULT:
		g_value_set_ulong (value, gkm_transaction_get_result (self));
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, prop_id, pspec);
		break;
	}
}

static void
gkm_transaction_class_init (GkmTransactionClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS (klass);

	gobject_class->dispose = gkm_transaction_dispose;
	gobject_class->finalize = gkm_transaction_finalize;
	gobject_class->set_property = gkm_transaction_set_property;
	gobject_class->get_property = gkm_transaction_get_property;

	klass->complete = gkm_transaction_real_complete;

	g_object_class_install_property (gobject_class, PROP_COMPLETED,
	           g_param_spec_boolean ("completed", "Completed", "Whether transaction is complete",
	                                 FALSE, G_PARAM_READABLE));

	g_object_class_install_property (gobject_class, PROP_FAILED,
	           g_param_spec_boolean ("failed", "Failed", "Whether transaction failed",
	                                 FALSE, G_PARAM_READABLE));

	g_object_class_install_property (gobject_class, PROP_RESULT,
	           g_param_spec_ulong ("result", "Result", "Result code for transaction",
	                               0, G_MAXULONG, CKR_OK, G_PARAM_READABLE));

	signals[COMPLETE] = g_signal_new ("complete", GKM_TYPE_TRANSACTION,
	                                  G_SIGNAL_RUN_LAST, G_STRUCT_OFFSET (GkmTransactionClass, complete),
	                                  complete_accumulator, NULL, gkm_marshal_BOOLEAN__VOID,
	                                  G_TYPE_BOOLEAN, 0, G_TYPE_NONE);
}

/* -----------------------------------------------------------------------------
 * PUBLIC
 */

GkmTransaction*
gkm_transaction_new (void)
{
	return g_object_new (GKM_TYPE_TRANSACTION, NULL);
}

void
gkm_transaction_add (GkmTransaction *self, gpointer object,
                     GkmTransactionFunc func, gpointer user_data)
{
	Complete *complete;

	g_return_if_fail (GKM_IS_TRANSACTION (self));
	g_return_if_fail (func);

	complete = g_slice_new0 (Complete);
	complete->func = func;
	if (object)
		complete->object = g_object_ref (object);
	complete->user_data = user_data;

	self->completes = g_list_prepend (self->completes, complete);
}

void
gkm_transaction_fail (GkmTransaction *self, CK_RV result)
{
	g_return_if_fail (GKM_IS_TRANSACTION (self));
	g_return_if_fail (!self->completed);
	g_return_if_fail (result != CKR_OK);
	g_return_if_fail (!self->failed);

	self->failed = TRUE;
	self->result = result;

	g_object_notify (G_OBJECT (self), "failed");
	g_object_notify (G_OBJECT (self), "result");
}

void
gkm_transaction_complete(GkmTransaction *self)
{
	gboolean critical = FALSE;

	g_return_if_fail (GKM_IS_TRANSACTION (self));
	g_return_if_fail (!self->completed);
	g_signal_emit (self, signals[COMPLETE], 0, &critical);
	g_assert (self->completed);

	if (!self->failed && critical) {
		g_warning ("transaction failed to commit, data may be lost");
		self->failed = TRUE;
		self->result = CKR_GENERAL_ERROR;
		g_object_notify (G_OBJECT (self), "failed");
		g_object_notify (G_OBJECT (self), "result");
	}
}

gboolean
gkm_transaction_get_completed (GkmTransaction *self)
{
	g_return_val_if_fail (GKM_IS_TRANSACTION (self), FALSE);
	return self->completed;
}

gboolean
gkm_transaction_get_failed (GkmTransaction *self)
{
	g_return_val_if_fail (GKM_IS_TRANSACTION (self), FALSE);
	return self->failed;
}

CK_RV
gkm_transaction_get_result (GkmTransaction *self)
{
	g_return_val_if_fail (GKM_IS_TRANSACTION (self), FALSE);
	return self->result;
}

void
gkm_transaction_write_file (GkmTransaction *self, const gchar *filename,
                            gconstpointer data, gsize n_data)
{
	gboolean exists;

	g_return_if_fail (GKM_IS_TRANSACTION (self));
	g_return_if_fail (filename);
	g_return_if_fail (data);
	g_return_if_fail (!gkm_transaction_get_failed (self));

	if (!begin_link_temporary_if_exists (self, filename, &exists))
		return;

	if (!exists) {
		if (!begin_new_file (self, filename))
			return;
	}

	/* Put data in the expected place */
	if (!write_to_file (filename, data, n_data)) {
		g_warning ("couldn't write to file: %s: %s", filename, g_strerror (errno));
		gkm_transaction_fail (self, CKR_DEVICE_ERROR);
	}
}

gchar*
gkm_transaction_unique_file (GkmTransaction *self, const gchar *directory,
                             const gchar *basename)
{
	gchar *ext;
	gchar *filename = NULL;
	gchar *base = NULL;
	gchar *result = NULL;
	gint seed = 1;
	int fd;

	g_return_val_if_fail (GKM_IS_TRANSACTION (self), NULL);
	g_return_val_if_fail (directory, NULL);
	g_return_val_if_fail (basename, NULL);
	g_return_val_if_fail (!gkm_transaction_get_failed (self), NULL);

	if (g_mkdir_with_parents (directory, S_IRWXU) < 0) {
		g_warning ("couldn't create directory: %s: %s", directory, g_strerror (errno));
		gkm_transaction_fail (self, CKR_DEVICE_ERROR);
		return NULL;
	}

	filename = g_build_filename (directory, basename, NULL);

	/* Write a zero byte file */
	fd = g_open (filename, O_RDONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
	if (fd != -1) {
		result = g_strdup (basename);

	/* Try to find a unique filename */
	} else if (errno == EEXIST) {
		base = g_strdup (basename);
		ext = strrchr (base, '.');
		if (ext != NULL)
			*(ext++) = '\0';

		do {
			g_free (result);
			result = g_strdup_printf ("%s_%d%s%s", base, seed++,
			                          ext ? "." : "", ext ? ext : "");

			g_free (filename);
			filename = g_build_filename (directory, result, NULL);
			fd = g_open (filename, O_RDONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);

		} while (seed < MAX_TRIES && fd == -1 && errno == EEXIST);
	}

	/* Something failed */
	if (fd == -1){
		g_warning ("couldn't open file: %s: %s", filename, g_strerror (errno));
		gkm_transaction_fail (self, CKR_DEVICE_ERROR);

	/* Success, just leave our zero byte file */
	} else {
		gkm_transaction_add (self, NULL, complete_new_file, filename);
		filename = NULL;
		close (fd);
	}

	g_free (filename);
	g_free (base);

	if (gkm_transaction_get_failed (self)) {
		g_free (result);
		result = NULL;
	}

	return result;
}

void
gkm_transaction_remove_file (GkmTransaction *self, const gchar *filename)
{
	gboolean exists;

	g_return_if_fail (GKM_IS_TRANSACTION (self));
	g_return_if_fail (filename);
	g_return_if_fail (!gkm_transaction_get_failed (self));

	if (!begin_link_temporary_if_exists (self, filename, &exists))
		return;

	/* Already gone? Job accomplished */
	if (!exists)
		return;

	/* If failure, temporary will automatically be removed */
	if (g_unlink (filename) < 0) {
		g_warning ("couldn't remove file: %s: %s", filename, g_strerror (errno));
		gkm_transaction_fail (self, CKR_DEVICE_ERROR);
	}
}

CK_RV
gkm_transaction_complete_and_unref (GkmTransaction *self)
{
	CK_RV rv;

	g_return_val_if_fail (GKM_IS_TRANSACTION (self), CKR_GENERAL_ERROR);

	gkm_transaction_complete (self);
	rv = gkm_transaction_get_result (self);
	g_object_unref (self);

	return rv;
}
