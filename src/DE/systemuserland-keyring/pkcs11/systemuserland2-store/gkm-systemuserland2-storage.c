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

#include "gkm-gnome2-file.h"
#include "gkm-gnome2-private-key.h"
#include "gkm-gnome2-public-key.h"
#include "gkm-gnome2-storage.h"

#include "gkm/gkm-certificate.h"
#define DEBUG_FLAG GKM_DEBUG_STORAGE
#include "gkm/gkm-debug.h"
#include "gkm/gkm-data-asn1.h"
#include "gkm/gkm-manager.h"
#include "gkm/gkm-module.h"
#include "gkm/gkm-secret.h"
#include "gkm/gkm-serializable.h"
#include "gkm/gkm-util.h"

#include "egg/dotlock.h"
#include "egg/egg-asn1x.h"
#include "egg/egg-asn1-defs.h"
#include "egg/egg-dn.h"
#include "egg/egg-error.h"
#include "egg/egg-hex.h"

#include "pkcs11/pkcs11i.h"

#include <glib/gstdio.h>

#include <sys/file.h>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

enum {
	PROP_0,
	PROP_MODULE,
	PROP_DIRECTORY,
	PROP_MANAGER,
	PROP_LOGIN
};

struct _GkmGnome2Storage {
	GkmStore parent;

	GkmModule *module;
	GkmManager *manager;

	/* Information about file data */
	gchar *directory;
	gchar *filename;
	GkmGnome2File *file;
	time_t last_mtime;
	GkmSecret *login;

	/* Mapping of objects loaded */
	GHashTable *object_to_identifier;
	GHashTable *identifier_to_object;

	/* Valid when in write state */
	GkmTransaction *transaction;
	gchar *write_path;
	gint write_fd;
	gint read_fd;
};

G_DEFINE_TYPE (GkmGnome2Storage, gkm_gnome2_storage, GKM_TYPE_STORE);

#define LOCK_TIMEOUT  4000  /*(4 seconds)*/

#define UNWANTED_IDENTIFIER_CHARS  ":/\\<>|\t\n\r\v "

static gchar*
name_for_subject (const guchar *subject,
                  gsize n_subject)
{
	GNode *asn;
	gchar *name;
	GBytes *bytes;

	g_assert (subject);
	g_assert (n_subject);

	bytes = g_bytes_new (subject, n_subject);
	asn = egg_asn1x_create_and_decode (pkix_asn1_tab, "Name", bytes);
	g_return_val_if_fail (asn != NULL, NULL);
	g_bytes_unref (bytes);

	name = egg_dn_read_part (egg_asn1x_node (asn, "rdnSequence", NULL), "CN");
	egg_asn1x_destroy (asn);

	return name;
}

static gchar*
identifier_for_object (GkmObject *object)
{
	GkmSerializableIface *serial;
	const gchar *ext;
	gchar *identifier;
	gchar *name = NULL;
	guchar *data;
	gsize n_data;

	g_assert (GKM_IS_OBJECT (object));
	g_assert (GKM_IS_SERIALIZABLE (object));

	/* Figure out the extension and prefix */
	serial = GKM_SERIALIZABLE_GET_INTERFACE (object);
	ext = serial->extension;
	g_return_val_if_fail (ext, NULL);

	/* First we try to use the CN of a subject */
	data = gkm_object_get_attribute_data (object, NULL, CKA_SUBJECT, &n_data);
	if (data && n_data)
		name = name_for_subject (data, n_data);
	g_free (data);

	/* Next we try hex encoding the ID */
	if (name == NULL) {
		data = gkm_object_get_attribute_data (object, NULL, CKA_ID, &n_data);
		if (data && n_data)
			name = egg_hex_encode (data, n_data);
		g_free (data);
	}

	/* Build up the identifier */
	identifier = g_strconcat (name, ext, NULL);
	g_strdelimit (identifier, UNWANTED_IDENTIFIER_CHARS, '_');

	g_free (name);
	return identifier;
}

static GType
type_from_extension (const gchar *extension)
{
	g_assert (extension);

	if (strcmp (extension, ".pkcs8") == 0)
		return GKM_TYPE_GNOME2_PRIVATE_KEY;
	else if (strcmp (extension, ".pub") == 0)
		return GKM_TYPE_GNOME2_PUBLIC_KEY;
	else if (strcmp (extension, ".cer") == 0)
		return GKM_TYPE_CERTIFICATE;

	return 0;
}


static GType
type_from_identifier (const gchar *identifier)
{
	const gchar *ext;

	g_assert (identifier);

	ext = strrchr (identifier, '.');
	if (ext == NULL)
		return 0;

	return type_from_extension (ext);
}

static dotlock_t
lock_and_open_file (const gchar *filename,
                    gint flags)
{
	dotlock_t lockh;
	gint fd = -1;

	/*
	 * In this function we don't actually put the object into a 'write' state,
	 * that's the callers job if necessary.
	 */

	fd = open (filename, O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);
	if (fd == -1) {
		g_message ("couldn't open store file: %s: %s",
		           filename, g_strerror (errno));
		return NULL;
	}

	lockh = dotlock_create (filename, 0);
	if (!lockh) {
		g_message ("couldn't create lock for store file: %s: %s",
		           filename, g_strerror (errno));
		close (fd);
		return NULL;
	}

	if (dotlock_take (lockh, LOCK_TIMEOUT)) {
		if (errno == EACCES)
			g_message ("couldn't write to store file:"
			           " %s: file is locked", filename);
		else
			g_message ("couldn't lock store file: %s: %s",
			           filename, g_strerror (errno));
		dotlock_destroy (lockh);
		close (fd);
		return NULL;
	}

	/* Successfully opened file */;
	dotlock_set_fd (lockh, fd);
	return lockh;
}

static gboolean
complete_lock_file (GkmTransaction *transaction, GObject *object, gpointer data)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (object);
	dotlock_t lockh = data;
	int fd = dotlock_get_fd (lockh);

	gkm_debug ("closing: %s", self->filename);

	/*
	 * Note that there is no error checking for release and
	 * destroy.  These functions will log errors anyway and we
	 * can't do much else than logging those errors.
	 */
	dotlock_release (lockh);
	dotlock_destroy (lockh);
	close (fd);

	/* Completed successfully */
	return TRUE;
}

static gint
begin_lock_file (GkmGnome2Storage *self, GkmTransaction *transaction)
{
	dotlock_t lockh;

	/*
	 * In this function we don't actually put the object into a 'write' state,
	 * that's the callers job if necessary.
	 */

	g_assert (GKM_IS_GNOME2_STORAGE (self));
	g_assert (GKM_IS_TRANSACTION (transaction));

	g_return_val_if_fail (!gkm_transaction_get_failed (transaction), -1);
	gkm_debug ("modifying: %s", self->filename);

	lockh = lock_and_open_file (self->filename, O_RDONLY | O_CREAT);
	if (!lockh) {
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		return -1;
	}

	/* Successfully opened file */;
	gkm_transaction_add (transaction, self, complete_lock_file, lockh);
	return dotlock_get_fd (lockh);
}

static gboolean
complete_write_state (GkmTransaction *transaction, GObject *object, gpointer unused)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (object);
	gboolean ret = TRUE;
	struct stat sb;

	g_return_val_if_fail (GKM_IS_GNOME2_STORAGE (object), FALSE);
	g_return_val_if_fail (GKM_IS_TRANSACTION (transaction), FALSE);
	g_return_val_if_fail (self->transaction == transaction, FALSE);

	if (!gkm_transaction_get_failed (transaction)) {
		/* Transaction succeeded, overwrite the old with the new */

		if (g_rename (self->write_path, self->filename) == -1) {
			g_warning ("couldn't rename temporary store file: %s", self->write_path);
			ret = FALSE;
		} else {
			if (fstat (self->write_fd, &sb) >= 0)
				self->last_mtime = sb.st_mtime;
		}
	} else {
		/* Transaction failed, remove temporary file */
		if (g_unlink (self->write_path) == -1)
			g_warning ("couldn't delete temporary store file: %s", self->write_path);
	}

	/* read_fd is closed by complete_lock_file */

	if (self->write_fd != -1)
		close (self->write_fd);
	self->write_fd = -1;

	g_free (self->write_path);
	self->write_path = NULL;

	g_object_unref (self->transaction);
	self->transaction = NULL;

	return ret;
}

static gboolean
begin_write_state (GkmGnome2Storage *self, GkmTransaction *transaction)
{
	g_assert (GKM_IS_GNOME2_STORAGE (self));
	g_assert (GKM_IS_TRANSACTION (transaction));

	g_return_val_if_fail (!gkm_transaction_get_failed (transaction), FALSE);

	/* Already in write state for this transaction? */
	if (self->transaction != NULL) {
		g_return_val_if_fail (self->transaction == transaction, FALSE);
		return TRUE;
	}

	/* Lock file for the transaction */
	self->read_fd = begin_lock_file (self, transaction);
	if (self->read_fd == -1)
		return FALSE;

	gkm_transaction_add (transaction, self, complete_write_state, NULL);
	self->transaction = g_object_ref (transaction);

	/* Open the new file */
	g_assert (self->write_fd == -1);
	self->write_path = g_strdup_printf ("%s.XXXXXX", self->filename);
	self->write_fd = g_mkstemp (self->write_path);
	if (self->write_fd == -1) {
		g_message ("couldn't open new temporary store file: %s: %s", self->write_path, g_strerror (errno));
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		return FALSE;
	}

	return TRUE;
}

static gboolean
complete_modification_state (GkmTransaction *transaction, GObject *object, gpointer unused)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (object);
	GkmDataResult res;

	if (!gkm_transaction_get_failed (transaction)) {
		res = gkm_gnome2_file_write_fd (self->file, self->write_fd, self->login);
		switch(res) {
		case GKM_DATA_FAILURE:
		case GKM_DATA_UNRECOGNIZED:
			g_warning ("couldn't write to temporary store file: %s", self->write_path);
			return FALSE;
		case GKM_DATA_LOCKED:
			g_warning ("couldn't encrypt temporary store file: %s", self->write_path);
			return FALSE;
		case GKM_DATA_SUCCESS:
			break;
		default:
			g_assert_not_reached ();
		}
	}

	return TRUE;
}

static gboolean
begin_modification_state (GkmGnome2Storage *self, GkmTransaction *transaction)
{
	GkmDataResult res;
	CK_RV rv = CKR_OK;

	/* Already in write state for this transaction? */
	if (self->transaction != NULL) {
		g_return_val_if_fail (self->transaction == transaction, FALSE);
		return TRUE;
	}

	if (!begin_write_state (self, transaction))
		return FALSE;

	/* See if file needs updating */
	res = gkm_gnome2_file_read_fd (self->file, self->read_fd, self->login);
	switch (res) {
	case GKM_DATA_FAILURE:
		g_message ("failure updating user store file: %s", self->filename);
		rv = CKR_FUNCTION_FAILED;
		break;
	case GKM_DATA_LOCKED:
		rv = CKR_USER_NOT_LOGGED_IN;
		break;
	case GKM_DATA_UNRECOGNIZED:
		g_message ("unrecognized or invalid user store file: %s", self->filename);
		rv = CKR_FUNCTION_FAILED;
		break;
	case GKM_DATA_SUCCESS:
		rv = CKR_OK;
		break;
	default:
		g_assert_not_reached ();
		break;
	}

	if (rv != CKR_OK) {
		gkm_transaction_fail (transaction, rv);
		return FALSE;
	}

	/* Write out the data once completed with modifications */
	gkm_transaction_add (transaction, self, complete_modification_state, NULL);

	return TRUE;
}

static void
take_object_ownership (GkmGnome2Storage *self, const gchar *identifier, GkmObject *object)
{
	gchar *str;

	g_assert (GKM_IS_GNOME2_STORAGE (self));
	g_assert (GKM_IS_OBJECT (object));

	g_assert (g_hash_table_lookup (self->identifier_to_object, identifier) == NULL);
	g_assert (g_hash_table_lookup (self->object_to_identifier, object) == NULL);

	str = g_strdup (identifier);
	object = g_object_ref (object);

	g_hash_table_replace (self->identifier_to_object, str, object);
	g_hash_table_replace (self->object_to_identifier, object, str);;

	g_object_set (object, "store", self, NULL);
	gkm_object_expose (object, TRUE);
}

static gboolean
check_object_hash (GkmGnome2Storage *self, const gchar *identifier, const guchar *data, gsize n_data)
{
	gconstpointer value;
	GkmDataResult res;
	gboolean result;
	gsize n_value;
	gchar *digest;

	g_assert (GKM_IS_GNOME2_STORAGE (self));
	g_assert (identifier);
	g_assert (data);

	digest = g_compute_checksum_for_data (G_CHECKSUM_SHA1, data, n_data);
	g_return_val_if_fail (digest, FALSE);

	res = gkm_gnome2_file_read_value (self->file, identifier, CKA_GNOME_INTERNAL_SHA1, &value, &n_value);
	g_return_val_if_fail (res == GKM_DATA_SUCCESS, FALSE);

	result = (strlen (digest) == n_value && memcmp (digest, value, n_value) == 0);
	g_free (digest);

	return result;
}

static void
store_object_hash (GkmGnome2Storage *self, GkmTransaction *transaction, const gchar *identifier,
                   const guchar *data, gsize n_data)
{
	GkmDataResult res;
	gchar *digest;

	g_assert (GKM_IS_GNOME2_STORAGE (self));
	g_assert (GKM_IS_TRANSACTION (transaction));
	g_assert (identifier);
	g_assert (data);

	digest = g_compute_checksum_for_data (G_CHECKSUM_SHA1, data, n_data);
	if (digest == NULL) {
		gkm_transaction_fail (transaction, CKR_GENERAL_ERROR);
		g_return_if_reached ();
	}

	res = gkm_gnome2_file_write_value (self->file, identifier, CKA_GNOME_INTERNAL_SHA1, digest, strlen (digest));
	g_free (digest);

	if (res != GKM_DATA_SUCCESS)
		gkm_transaction_fail (transaction, CKR_GENERAL_ERROR);
}

static void
data_file_entry_added (GkmGnome2File *store, const gchar *identifier, GkmGnome2Storage *self)
{
	GError *error = NULL;
	GkmObject *object;
	GBytes *bytes;
	gboolean ret;
	guchar *data;
	gsize n_data;
	GType type;
	gchar *path;

	g_return_if_fail (GKM_IS_GNOME2_STORAGE (self));
	g_return_if_fail (identifier);

	/* Already have this object? */
	object = g_hash_table_lookup (self->identifier_to_object, identifier);
	if (object != NULL)
		return;

	/* Figure out what type of object we're dealing with */
	type = type_from_identifier (identifier);
	if (type == 0) {
		g_warning ("don't know how to load file in user store: %s", identifier);
		return;
	}

	/* Read the file in */
	path = g_build_filename (self->directory, identifier, NULL);
	ret = g_file_get_contents (path, (gchar**)&data, &n_data, &error);
	g_free (path);

	if (ret == FALSE) {
		g_warning ("couldn't read file in user store: %s: %s", identifier,
		           egg_error_message (error));
		g_clear_error (&error);
		return;
	}

	/* Make sure that the object wasn't tampered with */
	if (!check_object_hash (self, identifier, data, n_data)) {
		g_message ("file in user store doesn't match hash: %s", identifier);
		g_free (data);
		return;
	}

	/* Create a new object for this identifier */
	object = g_object_new (type, "unique", identifier, "module", self->module,
	                       "manager", gkm_module_get_manager (self->module), NULL);
	g_return_if_fail (GKM_IS_SERIALIZABLE (object));
	g_return_if_fail (GKM_SERIALIZABLE_GET_INTERFACE (object)->extension);

	bytes = g_bytes_new_take (data, n_data);

	/* And load the data into it */
	if (gkm_serializable_load (GKM_SERIALIZABLE (object), self->login, bytes))
		take_object_ownership (self, identifier, object);
	else
		g_message ("failed to load file in user store: %s", identifier);

	g_bytes_unref (bytes);
	g_object_unref (object);
}

static void
data_file_entry_changed (GkmGnome2File *store, const gchar *identifier, CK_ATTRIBUTE_TYPE type, GkmGnome2Storage *self)
{
	GkmObject *object;

	g_return_if_fail (GKM_IS_GNOME2_STORAGE (self));
	g_return_if_fail (identifier);

	object = g_hash_table_lookup (self->identifier_to_object, identifier);
	if (object != NULL)
		gkm_object_notify_attribute (object, type);
}

static void
data_file_entry_removed (GkmGnome2File *store, const gchar *identifier, GkmGnome2Storage *self)
{
	GkmObject *object;

	g_return_if_fail (GKM_IS_GNOME2_STORAGE (self));
	g_return_if_fail (identifier);

	object = g_hash_table_lookup (self->identifier_to_object, identifier);
	if (object != NULL) {
		g_object_set (object, "store", NULL, NULL);

		/* Unrefs and also disposes the object, which unregisters from manager*/
		g_hash_table_remove (self->identifier_to_object, identifier);
		g_hash_table_remove (self->object_to_identifier, object);
	}
}

static void
relock_object (GkmGnome2Storage *self, GkmTransaction *transaction, const gchar *path,
               const gchar *identifier, GkmSecret *old_login, GkmSecret *new_login)
{
	GError *error = NULL;
	GkmObject *object;
	GBytes *bytes;
	gpointer data;
	gsize n_data;
	GType type;

	g_assert (GKM_IS_GNOME2_STORAGE (self));
	g_assert (GKM_IS_TRANSACTION (transaction));
	g_assert (identifier);
	g_assert (path);

	g_assert (!gkm_transaction_get_failed (transaction));

	/* Figure out the type of object */
	type = type_from_identifier (identifier);
	if (type == 0) {
		g_warning ("don't know how to relock file in user store: %s", identifier);
		gkm_transaction_fail (transaction, CKR_GENERAL_ERROR);
		return;
	}

	/* Create a dummy object for this identifier */
	object = g_object_new (type, "unique", identifier, "module", self->module, NULL);
	if (!GKM_IS_SERIALIZABLE (object)) {
		g_warning ("cannot relock unserializable object for file in user store: %s", identifier);
		gkm_transaction_fail (transaction, CKR_GENERAL_ERROR);
		return;
	}

	/* Read in the data for the object */
	if (!g_file_get_contents (path, (gchar**)&data, &n_data, &error)) {
		g_message ("couldn't load file in user store in order to relock: %s: %s", identifier,
		           egg_error_message (error));
		g_clear_error (&error);
		g_object_unref (object);
		gkm_transaction_fail (transaction, CKR_GENERAL_ERROR);
		return;
	}

	/* Make sure the data matches the hash */
	if (!check_object_hash (self, identifier, data, n_data)) {
		g_message ("file in data store doesn't match hash: %s", identifier);
		gkm_transaction_fail (transaction, CKR_GENERAL_ERROR);
		g_free (data);
		return;
	}

	bytes = g_bytes_new_take (data, n_data);

	/* Load it into our temporary object */
	if (!gkm_serializable_load (GKM_SERIALIZABLE (object), old_login, bytes)) {
		g_message ("unrecognized or invalid user store file: %s", identifier);
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		g_bytes_unref (bytes);
		g_object_unref (object);
		return;
	}

	g_bytes_unref (bytes);

	/* Read it out of our temporary object */
	bytes = gkm_serializable_save (GKM_SERIALIZABLE (object), new_login);
	if (bytes == NULL) {
		g_warning ("unable to serialize data with new login: %s", identifier);
		gkm_transaction_fail (transaction, CKR_GENERAL_ERROR);
		g_object_unref (object);
		g_free (data);
		return;
	}

	g_object_unref (object);

	/* And write it back out to the file */
	gkm_transaction_write_file (transaction, path, data, n_data);

	/* Create and save the hash here */
	if (!gkm_transaction_get_failed (transaction))
		store_object_hash (self, transaction, identifier, data, n_data);

	g_bytes_unref (bytes);
}

typedef struct _RelockArgs {
	GkmGnome2Storage *self;
	GkmTransaction *transaction;
	GkmSecret *old_login;
	GkmSecret *new_login;
} RelockArgs;

static void
relock_each_object (GkmGnome2File *file, const gchar *identifier, gpointer data)
{
	RelockArgs *args = data;
	gchar *path;
	guint section;

	g_assert (GKM_IS_GNOME2_STORAGE (args->self));
	if (gkm_transaction_get_failed (args->transaction))
		return;

	if (!gkm_gnome2_file_lookup_entry (file, identifier, &section))
		g_return_if_reached ();

	/* Only operate on private files */
	if (section != GKM_GNOME2_FILE_SECTION_PRIVATE)
		return;

	path = g_build_filename (args->self->directory, identifier, NULL);
	relock_object (args->self, args->transaction, path, identifier, args->old_login, args->new_login);
	g_free (path);
}

static CK_RV
refresh_with_login (GkmGnome2Storage *self, GkmSecret *login)
{
	GkmDataResult res;
	dotlock_t lockh;
	struct stat sb;
	CK_RV rv;
	int fd;

	g_assert (GKM_GNOME2_STORAGE (self));
	gkm_debug ("refreshing: %s", self->filename);

	/* Open the file for reading */
	lockh = lock_and_open_file (self->filename, O_RDONLY);
	if (!lockh) {
		/* No file, no worries */
		if (errno == ENOENT)
			return login ? CKR_USER_PIN_NOT_INITIALIZED : CKR_OK;
		g_message ("couldn't open store file: %s: %s", self->filename, g_strerror (errno));
		return CKR_FUNCTION_FAILED;
	}

	fd = dotlock_get_fd (lockh);

	/* Try and update the last read time */
	if (fstat (fd, &sb) >= 0)
		self->last_mtime = sb.st_mtime;

	res = gkm_gnome2_file_read_fd (self->file, fd, login);
	switch (res) {
	case GKM_DATA_FAILURE:
		g_message ("failure reading from file: %s", self->filename);
		rv = CKR_FUNCTION_FAILED;
		break;
	case GKM_DATA_LOCKED:
		rv = CKR_USER_NOT_LOGGED_IN;
		break;
	case GKM_DATA_UNRECOGNIZED:
		g_message ("unrecognized or invalid user store file: %s", self->filename);
		rv = CKR_FUNCTION_FAILED;
		break;
	case GKM_DATA_SUCCESS:
		rv = CKR_OK;
		break;
	default:
		g_assert_not_reached ();
		break;
	}

	/* Force a reread on next write */
	if (rv == CKR_FUNCTION_FAILED)
		self->last_mtime = 0;

	gkm_debug ("closing: %s", self->filename);

	/*
	 * Note that there is no error checking for release and
	 * destroy.  These functions will log errors anyway and we
	 * can't do much else than logging those errors.
	 */
	dotlock_release (lockh);
	dotlock_destroy (lockh);
	close (fd);

	return rv;
}

/* -----------------------------------------------------------------------------
 * OBJECT
 */

static CK_RV
gkm_gnome2_storage_real_read_value (GkmStore *base, GkmObject *object, CK_ATTRIBUTE_PTR attr)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (base);
	const gchar *identifier;
	GkmDataResult res;
	gconstpointer value;
	gsize n_value;
	CK_RV rv;

	g_return_val_if_fail (GKM_IS_GNOME2_STORAGE (self), CKR_GENERAL_ERROR);
	g_return_val_if_fail (GKM_IS_OBJECT (object), CKR_GENERAL_ERROR);
	g_return_val_if_fail (attr, CKR_GENERAL_ERROR);

	if (self->last_mtime == 0) {
		rv = gkm_gnome2_storage_refresh (self);
		if (rv != CKR_OK)
			return rv;
	}

	identifier = g_hash_table_lookup (self->object_to_identifier, object);
	if (!identifier) {
		gkm_debug ("CKR_ATTRIBUTE_TYPE_INVALID: object not stored in gnome2 storage");
		return CKR_ATTRIBUTE_TYPE_INVALID;
	}

	res = gkm_gnome2_file_read_value (self->file, identifier, attr->type, &value, &n_value);
	switch (res) {
	case GKM_DATA_FAILURE:
		g_return_val_if_reached (CKR_GENERAL_ERROR);
	case GKM_DATA_UNRECOGNIZED:
		gkm_debug ("CKR_ATTRIBUTE_TYPE_INVALID: attribute %s not present",
		           gkm_log_attr_type (attr->type));
		return CKR_ATTRIBUTE_TYPE_INVALID;
	case GKM_DATA_LOCKED:
		return CKR_USER_NOT_LOGGED_IN;
	case GKM_DATA_SUCCESS:
		/* Yes, we don't fill a buffer, just return pointer */
		attr->pValue = (CK_VOID_PTR)value;
		attr->ulValueLen = n_value;
		return CKR_OK;
	default:
		g_assert_not_reached ();
	}
}

static void
gkm_gnome2_storage_real_write_value (GkmStore *base, GkmTransaction *transaction, GkmObject *object, CK_ATTRIBUTE_PTR attr)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (base);
	const gchar *identifier;
	GkmDataResult res;
	CK_RV rv;

	g_return_if_fail (GKM_IS_GNOME2_STORAGE (self));
	g_return_if_fail (GKM_IS_OBJECT (object));
	g_return_if_fail (GKM_IS_TRANSACTION (transaction));
	g_return_if_fail (!gkm_transaction_get_failed (transaction));
	g_return_if_fail (attr);

	if (!begin_modification_state (self, transaction))
		return;

	identifier = g_hash_table_lookup (self->object_to_identifier, object);
	if (!identifier) {
		gkm_transaction_fail (transaction, CKR_ATTRIBUTE_READ_ONLY);
		return;
	}

	res = gkm_gnome2_file_write_value (self->file, identifier, attr->type, attr->pValue, attr->ulValueLen);
	switch (res) {
	case GKM_DATA_FAILURE:
		rv = CKR_FUNCTION_FAILED;
		break;
	case GKM_DATA_UNRECOGNIZED:
		rv = CKR_ATTRIBUTE_READ_ONLY;
		break;
	case GKM_DATA_LOCKED:
		rv = CKR_USER_NOT_LOGGED_IN;
		break;
	case GKM_DATA_SUCCESS:
		rv = CKR_OK;
		break;
	default:
		g_assert_not_reached ();
	}

	if (rv != CKR_OK)
		gkm_transaction_fail (transaction, rv);
}

static GObject*
gkm_gnome2_storage_constructor (GType type, guint n_props, GObjectConstructParam *props)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (G_OBJECT_CLASS (gkm_gnome2_storage_parent_class)->constructor(type, n_props, props));
	CK_ATTRIBUTE attr;

	g_return_val_if_fail (self, NULL);

	g_return_val_if_fail (self->directory, NULL);
	self->filename = g_build_filename (self->directory, "user.keystore", NULL);

	g_return_val_if_fail (self->manager, NULL);
	g_return_val_if_fail (self->module, NULL);

	/* Register store attributes */
	attr.type = CKA_LABEL;
	attr.pValue = "";
	attr.ulValueLen = 0;

	gkm_store_register_schema (GKM_STORE (self), &attr, NULL, 0);

	return G_OBJECT (self);
}

static void
gkm_gnome2_storage_init (GkmGnome2Storage *self)
{
	self->file = gkm_gnome2_file_new ();
	g_signal_connect (self->file, "entry-added", G_CALLBACK (data_file_entry_added), self);
	g_signal_connect (self->file, "entry-changed", G_CALLBACK (data_file_entry_changed), self);
	g_signal_connect (self->file, "entry-removed", G_CALLBACK (data_file_entry_removed), self);

	/* Each one owns the key and contains weak ref to other's key as its value */
	self->object_to_identifier = g_hash_table_new_full (g_direct_hash, g_direct_equal, gkm_util_dispose_unref, NULL);
	self->identifier_to_object = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);

	self->read_fd = -1;
	self->write_fd = -1;
}

static void
gkm_gnome2_storage_dispose (GObject *obj)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (obj);

	if (self->manager)
		g_object_unref (self->manager);
	self->manager = NULL;

	if (self->login)
		g_object_unref (self->login);
	self->login = NULL;

	g_signal_handlers_disconnect_by_func (self->file, data_file_entry_added, self);
	g_signal_handlers_disconnect_by_func (self->file, data_file_entry_changed, self);
	g_signal_handlers_disconnect_by_func (self->file, data_file_entry_removed, self);

	g_hash_table_remove_all (self->object_to_identifier);
	g_hash_table_remove_all (self->identifier_to_object);

	G_OBJECT_CLASS (gkm_gnome2_storage_parent_class)->dispose (obj);
}

static void
gkm_gnome2_storage_finalize (GObject *obj)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (obj);

	g_assert (self->file);
	g_object_unref (self->file);
	self->file = NULL;

	g_free (self->filename);
	self->filename = NULL;

	g_assert (self->directory);
	g_free (self->directory);
	self->directory = NULL;

	g_assert (self->object_to_identifier);
	g_hash_table_destroy (self->object_to_identifier);
	g_hash_table_destroy (self->identifier_to_object);

	G_OBJECT_CLASS (gkm_gnome2_storage_parent_class)->finalize (obj);
}

static void
gkm_gnome2_storage_set_property (GObject *obj, guint prop_id, const GValue *value,
                           GParamSpec *pspec)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (obj);

	switch (prop_id) {
	case PROP_DIRECTORY:
		g_return_if_fail (!self->directory);
		self->directory = g_value_dup_string (value);
		g_return_if_fail (self->directory);
		break;
	case PROP_MODULE:
		g_return_if_fail (!self->module);
		self->module = g_value_get_object (value);
		break;
	case PROP_MANAGER:
		g_return_if_fail (!self->manager);
		self->manager = g_value_dup_object (value);
		g_return_if_fail (self->manager);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, prop_id, pspec);
		break;
	}
}

static void
gkm_gnome2_storage_get_property (GObject *obj, guint prop_id, GValue *value,
                               GParamSpec *pspec)
{
	GkmGnome2Storage *self = GKM_GNOME2_STORAGE (obj);

	switch (prop_id) {
	case PROP_DIRECTORY:
		g_value_set_string (value, gkm_gnome2_storage_get_directory (self));
		break;
	case PROP_MODULE:
		g_value_set_object (value, self->module);
		break;
	case PROP_MANAGER:
		g_value_set_object (value, gkm_gnome2_storage_get_manager (self));
		break;
	case PROP_LOGIN:
		g_value_set_object (value, gkm_gnome2_storage_get_login (self));
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (obj, prop_id, pspec);
		break;
	}
}

static void
gkm_gnome2_storage_class_init (GkmGnome2StorageClass *klass)
{
	GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
	GkmStoreClass *store_class = GKM_STORE_CLASS (klass);

	gobject_class->constructor = gkm_gnome2_storage_constructor;
	gobject_class->dispose = gkm_gnome2_storage_dispose;
	gobject_class->finalize = gkm_gnome2_storage_finalize;
	gobject_class->set_property = gkm_gnome2_storage_set_property;
	gobject_class->get_property = gkm_gnome2_storage_get_property;

	store_class->read_value = gkm_gnome2_storage_real_read_value;
	store_class->write_value = gkm_gnome2_storage_real_write_value;

	g_object_class_install_property (gobject_class, PROP_DIRECTORY,
	           g_param_spec_string ("directory", "Storage Directory", "Directory for storage",
	                                NULL, G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

	g_object_class_install_property (gobject_class, PROP_MODULE,
	           g_param_spec_object ("module", "Module", "Module for objects",
	                                GKM_TYPE_MODULE, G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

	g_object_class_install_property (gobject_class, PROP_MANAGER,
	           g_param_spec_object ("manager", "Object Manager", "Object Manager",
	                                GKM_TYPE_MANAGER, G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY));

	g_object_class_install_property (gobject_class, PROP_LOGIN,
	           g_param_spec_object ("login", "Login", "Login used to unlock",
	                                GKM_TYPE_SECRET, G_PARAM_READABLE));
}

/* -----------------------------------------------------------------------------
 * PUBLIC
 */

GkmGnome2Storage*
gkm_gnome2_storage_new (GkmModule *module, const gchar *directory)
{
	GkmManager *manager;

	g_return_val_if_fail (GKM_IS_MODULE (module), NULL);
	g_return_val_if_fail (directory, NULL);

	manager = gkm_module_get_manager (module);
	g_return_val_if_fail (GKM_IS_MANAGER (manager), NULL);

	return g_object_new (GKM_TYPE_GNOME2_STORAGE,
	                     "module", module,
	                     "manager", manager,
	                     "directory", directory,
	                     NULL);
}

CK_RV
gkm_gnome2_storage_refresh (GkmGnome2Storage *self)
{
	g_return_val_if_fail (GKM_GNOME2_STORAGE (self), CKR_GENERAL_ERROR);
	return refresh_with_login (self, self->login);
}

void
gkm_gnome2_storage_create (GkmGnome2Storage *self, GkmTransaction *transaction, GkmObject *object)
{
	gboolean is_private;
	GkmDataResult res;
	gchar *identifier;
	GBytes *data;
	gchar *path;

	g_return_if_fail (GKM_IS_GNOME2_STORAGE (self));
	g_return_if_fail (GKM_IS_TRANSACTION (transaction));
	g_return_if_fail (!gkm_transaction_get_failed (transaction));
	g_return_if_fail (GKM_IS_OBJECT (object));

	/* Make sure we haven't already stored it */
	identifier = g_hash_table_lookup (self->object_to_identifier, object);
	g_return_if_fail (identifier == NULL);

	/* Double check that the object is in fact serializable */
	if (!GKM_IS_SERIALIZABLE (object)) {
		g_warning ("can't store object of type '%s' on token", G_OBJECT_TYPE_NAME (object));
		gkm_transaction_fail (transaction, CKR_GENERAL_ERROR);
		g_return_if_reached ();
	}

	/* Figure out whether this is a private object */
	if (!gkm_object_get_attribute_boolean (object, NULL, CKA_PRIVATE, &is_private))
		is_private = FALSE;

	/* Can't serialize private if we're not unlocked */
	if (is_private && !self->login) {
		gkm_transaction_fail (transaction, CKR_USER_NOT_LOGGED_IN);
		return;
	}

	/* Hook ourselves into the transaction */
	if (!begin_modification_state (self, transaction))
		return;

	/* Create an identifier guaranteed unique by this transaction */
	identifier = identifier_for_object (object);
	if (gkm_gnome2_file_unique_entry (self->file, &identifier) != GKM_DATA_SUCCESS) {
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		g_return_if_reached ();
	}

	/* We don't want to get signals about this item being added */
	g_signal_handlers_block_by_func (self->file, data_file_entry_added, self);
	g_signal_handlers_block_by_func (self->file, data_file_entry_changed, self);

	res = gkm_gnome2_file_create_entry (self->file, identifier,
	                                  is_private ? GKM_GNOME2_FILE_SECTION_PRIVATE : GKM_GNOME2_FILE_SECTION_PUBLIC);

	g_signal_handlers_unblock_by_func (self->file, data_file_entry_added, self);
	g_signal_handlers_unblock_by_func (self->file, data_file_entry_changed, self);

	switch(res) {
	case GKM_DATA_FAILURE:
	case GKM_DATA_UNRECOGNIZED:
		g_free (identifier);
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		return;
	case GKM_DATA_LOCKED:
		g_free (identifier);
		gkm_transaction_fail (transaction, CKR_USER_NOT_LOGGED_IN);
		return;
	case GKM_DATA_SUCCESS:
		break;
	default:
		g_assert_not_reached ();
	}

	/* Serialize the object in question */
	data = gkm_serializable_save (GKM_SERIALIZABLE (object), is_private ? self->login : NULL);

	if (data == NULL) {
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		g_return_if_reached ();
	}

	path = g_build_filename (self->directory, identifier, NULL);
	gkm_transaction_write_file (transaction, path,
	                            g_bytes_get_data (data, NULL),
	                            g_bytes_get_size (data));

	/* Make sure we write in the object hash */
	if (!gkm_transaction_get_failed (transaction))
		store_object_hash (self, transaction, identifier,
		                   g_bytes_get_data (data, NULL),
		                   g_bytes_get_size (data));

	/* Now we decide to own the object */
	if (!gkm_transaction_get_failed (transaction))
		take_object_ownership (self, identifier, object);

	g_free (identifier);
	g_free (path);
	g_bytes_unref (data);
}

void
gkm_gnome2_storage_destroy (GkmGnome2Storage *self, GkmTransaction *transaction, GkmObject *object)
{
	GkmDataResult res;
	gchar *identifier;
	gchar *path;

	g_return_if_fail (GKM_IS_GNOME2_STORAGE (self));
	g_return_if_fail (GKM_IS_TRANSACTION (transaction));
	g_return_if_fail (!gkm_transaction_get_failed (transaction));
	g_return_if_fail (object);

	/* Lookup the object identifier */
	identifier = g_hash_table_lookup (self->object_to_identifier, object);
	g_return_if_fail (identifier);

	if (!begin_modification_state (self, transaction))
		return;

	/* First actually delete the file */
	path = g_build_filename (self->directory, identifier, NULL);
	gkm_transaction_remove_file (transaction, path);
	g_free (path);

	if (gkm_transaction_get_failed (transaction))
		return;

	/* Now delete the entry from our store */
	res = gkm_gnome2_file_destroy_entry (self->file, identifier);
	switch(res) {
	case GKM_DATA_FAILURE:
	case GKM_DATA_UNRECOGNIZED:
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		return;
	case GKM_DATA_LOCKED:
		gkm_transaction_fail (transaction, CKR_USER_NOT_LOGGED_IN);
		return;
	case GKM_DATA_SUCCESS:
		break;
	default:
		g_assert_not_reached ();
	}

	/* Actual removal of object happened as a callback above */
	g_return_if_fail (g_hash_table_lookup (self->object_to_identifier, object) == NULL);
}

void
gkm_gnome2_storage_relock (GkmGnome2Storage *self, GkmTransaction *transaction,
                         GkmSecret *old_login, GkmSecret *new_login)
{
	GkmGnome2File *file;
	GkmDataResult res;
	RelockArgs args;

	g_return_if_fail (GKM_IS_GNOME2_STORAGE (self));
	g_return_if_fail (GKM_IS_TRANSACTION (transaction));

	/* Reload the file with the old password and start transaction */
	if (!begin_write_state (self, transaction))
		return;

	file = gkm_gnome2_file_new ();

	/* Read in from the old file */
	res = gkm_gnome2_file_read_fd (file, self->read_fd, old_login);
	switch(res) {
	case GKM_DATA_FAILURE:
	case GKM_DATA_UNRECOGNIZED:
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		return;
	case GKM_DATA_LOCKED:
		gkm_transaction_fail (transaction, CKR_PIN_INCORRECT);
		return;
	case GKM_DATA_SUCCESS:
		break;
	default:
		g_assert_not_reached ();
	}

	/* Write out to new path as new file */
	res = gkm_gnome2_file_write_fd (file, self->write_fd, new_login);
	switch(res) {
	case GKM_DATA_FAILURE:
	case GKM_DATA_UNRECOGNIZED:
		gkm_transaction_fail (transaction, CKR_FUNCTION_FAILED);
		return;
	case GKM_DATA_LOCKED:
		gkm_transaction_fail (transaction, CKR_PIN_INCORRECT);
		return;
	case GKM_DATA_SUCCESS:
		break;
	default:
		g_assert_not_reached ();
	}

	/* Now go through all objects in the file, and load and reencode them */
	args.transaction = transaction;
	args.old_login = old_login;
	args.new_login = new_login;
	args.self = self;
	gkm_gnome2_file_foreach_entry (file, relock_each_object, &args);

	if (!gkm_transaction_get_failed (transaction) && self->login) {
		if (new_login)
			g_object_ref (new_login);
		g_object_unref (self->login);
		self->login = new_login;
		g_object_notify (G_OBJECT (self), "login");
	}

	g_object_unref (file);
}

CK_RV
gkm_gnome2_storage_unlock (GkmGnome2Storage *self, GkmSecret *login)
{
	CK_RV rv;

	g_return_val_if_fail (GKM_IS_GNOME2_STORAGE (self), CKR_GENERAL_ERROR);
	g_return_val_if_fail (!self->transaction, CKR_GENERAL_ERROR);

	if (self->login)
		return CKR_USER_ALREADY_LOGGED_IN;

	self->login = login;

	rv = refresh_with_login (self, login);
	if (rv == CKR_USER_NOT_LOGGED_IN)
		rv = CKR_PIN_INCORRECT;

	/* Take on new login for good */
	if (rv == CKR_OK) {
		g_assert (self->login == login);
		if (self->login)
			g_object_ref (self->login);
		g_object_notify (G_OBJECT (self), "login");

	/* Failed, so keep our previous NULL login */
	} else {
		self->login = NULL;
	}

	return rv;
}

CK_RV
gkm_gnome2_storage_lock (GkmGnome2Storage *self)
{
	GkmSecret *prev;
	CK_RV rv;

	g_return_val_if_fail (GKM_IS_GNOME2_STORAGE (self), CKR_GENERAL_ERROR);
	g_return_val_if_fail (!self->transaction, CKR_GENERAL_ERROR);

	if (!self->login)
		return CKR_USER_NOT_LOGGED_IN;

	/* While loading set new NULL login */
	prev = self->login;
	self->login = NULL;

	rv = refresh_with_login (self, NULL);

	/* Take on new login for good */
	if (rv == CKR_OK) {
		g_object_unref (prev);
		g_assert (self->login == NULL);
		g_object_notify (G_OBJECT (self), "login");

	/* Failed so revert to previous login */
	} else {
		self->login = prev;
	}

	return rv;
}

GkmManager*
gkm_gnome2_storage_get_manager (GkmGnome2Storage *self)
{
	g_return_val_if_fail (GKM_IS_GNOME2_STORAGE (self), NULL);
	return self->manager;
}

const gchar*
gkm_gnome2_storage_get_directory (GkmGnome2Storage *self)
{
	g_return_val_if_fail (GKM_IS_GNOME2_STORAGE (self), NULL);
	return self->directory;
}

GkmSecret*
gkm_gnome2_storage_get_login (GkmGnome2Storage *self)
{
	g_return_val_if_fail (GKM_IS_GNOME2_STORAGE (self), NULL);
	return self->login;
}

gulong
gkm_gnome2_storage_token_flags (GkmGnome2Storage *self)
{
	gulong flags = 0;
	CK_RV rv;

	/* We don't changing SO logins, so always initialized */
	flags |= CKF_TOKEN_INITIALIZED | CKF_LOGIN_REQUIRED;

	/* No file has been loaded yet? */
	if (self->last_mtime == 0) {
		rv = gkm_gnome2_storage_refresh (self);
		if (rv == CKR_USER_PIN_NOT_INITIALIZED)
			flags |= CKF_USER_PIN_TO_BE_CHANGED;
		else if (rv != CKR_OK)
			g_return_val_if_reached (flags);
	}

	/* No private stuff in the file? */
	if (gkm_gnome2_file_have_section (self->file, GKM_GNOME2_FILE_SECTION_PRIVATE))
		flags |= CKF_USER_PIN_INITIALIZED;

	return flags;
}
