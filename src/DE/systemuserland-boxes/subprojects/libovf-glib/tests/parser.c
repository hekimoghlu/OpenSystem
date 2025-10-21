/*
 * Copyright (C) 2016 Kalev Lember <klember@redhat.com>
 *
 * Licensed under the GNU Lesser General Public License Version 2.1
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
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#include <glib/gstdio.h>
#include "govf-disk.h"
#include "govf-package.h"

static void
test_init_parser (void)
{
	g_autoptr(GovfPackage) ovf_package = NULL;

	ovf_package = govf_package_new ();
	g_assert (GOVF_IS_PACKAGE (ovf_package));
}

static void
test_missing_sections (void)
{
	g_autoptr(GError) error = NULL;
	g_autoptr(GovfPackage) ovf_package = NULL;
	gchar data[] =
"<?xml version=\"1.0\"?>"
"<Envelope ovf:version=\"1.0\" xml:lang=\"en-US\" xmlns=\"http://schemas.dmtf.org/ovf/envelope/1\" xmlns:ovf=\"http://schemas.dmtf.org/ovf/envelope/1\">"
"  <VirtualSystem ovf:id=\"Fedora 23\">"
"  </VirtualSystem>"
"</Envelope>";

	ovf_package = govf_package_new ();
	govf_package_load_from_data (ovf_package, data, -1, &error);
	g_assert_error (error, GOVF_PACKAGE_ERROR, GOVF_PACKAGE_ERROR_XML);
}

static void
test_load_valid_ovf (void)
{
	g_autoptr(GError) error = NULL;
	g_autoptr(GovfPackage) ovf_package = NULL;

	ovf_package = govf_package_new ();
	govf_package_load_from_file (ovf_package,
	                             g_build_filename ("tests", "Fedora_23.ovf", NULL),
	                             &error);
	g_assert_no_error (error);
}

static void
test_load_valid_ova (void)
{
	g_autoptr(GError) error = NULL;
	g_autoptr(GovfPackage) ovf_package = NULL;

	ovf_package = govf_package_new ();
	govf_package_load_from_ova_file (ovf_package,
	                                 "Fedora_23.ova",
	                                 NULL, NULL, NULL);
	g_assert_no_error (error);
}

static void
test_save_ovf (void)
{
	g_autofree gchar *contents = NULL;
	g_autofree gchar *filename = NULL;
	g_autofree gchar *tmp_dir = NULL;
	g_autoptr(GError) error = NULL;
	g_autoptr(GovfPackage) ovf_package = NULL;
	gsize length = 0;

	/* first load in an ovf file */
	ovf_package = govf_package_new ();
	govf_package_load_from_file (ovf_package,
	                             g_build_filename ("tests", "Fedora_23.ovf", NULL),
	                             &error);
	g_assert_no_error (error);

	tmp_dir = g_dir_make_tmp ("libgovf-test-XXXXXX", &error);
	g_assert_no_error (error);

	/* save it back to disk */
	filename = g_build_filename (tmp_dir, "Fedora_23.ovf", NULL);
	govf_package_save_file (ovf_package, filename, &error);
	g_assert_no_error (error);

	/* make sure it's not 0 bytes */
	g_file_get_contents (filename, &contents, &length, &error);
	g_assert (length > 0);
	g_assert_no_error (error);

	g_unlink (filename);
	g_rmdir (tmp_dir);
}

static void
test_get_disks (void)
{
	g_autoptr(GError) error = NULL;
	g_autoptr(GPtrArray) ovf_disks = NULL;
	g_autoptr(GovfPackage) ovf_package = NULL;
	const gchar *disk_id;

	ovf_package = govf_package_new ();
	govf_package_load_from_file (ovf_package,
	                             g_build_filename ("tests", "Fedora_23.ovf", NULL),
	                             &error);
	g_assert_no_error (error);

	ovf_disks = govf_package_get_disks (ovf_package);
	g_assert (ovf_disks != NULL);
	g_assert (ovf_disks->len == 1);

	disk_id = govf_disk_get_disk_id ((GovfDisk *) g_ptr_array_index (ovf_disks, 0));
	g_assert_cmpstr (disk_id, ==, "vmdisk2");
}

static void
on_govf_package_extracted (GObject      *source_object,
                           GAsyncResult *result,
                           gpointer      user_data)
{
	g_autofree gchar *filename = NULL;
	g_autofree gchar *tmp_dir = NULL;
	g_autoptr(GError) error = NULL;
	g_autoptr(GPtrArray) ovf_disks = NULL;
	g_autoptr(GovfPackage) ovf_package = (GovfPackage*)source_object;

	ovf_disks = govf_package_get_disks (ovf_package);
	g_assert (ovf_disks != NULL);
	g_assert (ovf_disks->len == 1);

	tmp_dir = g_dir_make_tmp ("libgovf-test-XXXXXX", &error);
	g_assert_no_error (error);

	filename = g_build_filename (tmp_dir, "extracted disk.vmdk", NULL);
	govf_package_extract_disk (ovf_package,
	                           (GovfDisk *) g_ptr_array_index (ovf_disks, 0),
	                           filename,
	                           NULL, NULL, NULL);
	g_assert_no_error (error);
	g_assert (g_file_test (filename, G_FILE_TEST_EXISTS));

	g_unlink (filename);
	g_rmdir (tmp_dir);
}

static void
test_extract_disk (void)
{
	g_autoptr(GovfPackage) ovf_package = NULL;

	ovf_package = govf_package_new ();
	govf_package_load_from_ova_file (ovf_package,
	                                 g_build_filename ("tests", "Fedora_23.ovf", NULL),
	                                 NULL,
	                                 (GAsyncReadyCallback)on_govf_package_extracted,
	                                 NULL);
}

int
main (int   argc,
      char *argv[])
{
	g_test_init (&argc, &argv, NULL);

	g_test_add_func ("/parser/init-parser", test_init_parser);
	g_test_add_func ("/parser/missing-sections", test_missing_sections);
	g_test_add_func ("/parser/load-valid-ovf", test_load_valid_ovf);
	g_test_add_func ("/parser/load-valid-ova", test_load_valid_ova);
	g_test_add_func ("/parser/save-ovf", test_save_ovf);
	g_test_add_func ("/parser/get-disks", test_get_disks);
	g_test_add_func ("/parser/extract-disk", test_extract_disk);

	return g_test_run ();
}
