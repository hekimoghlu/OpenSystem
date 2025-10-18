/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 8; tab-width: 8 -*- */
/* dump-data-file.c: Dump a gkm data file

   Copyright (C) 2009 Stefan Walter

   The Gnome Keyring Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The Gnome Keyring Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the Gnome Library; see the file COPYING.LIB.  If not,
   <http://www.gnu.org/licenses/>.

   Author: Stef Walter <stef@memberwebs.com>
*/

#include "config.h"

#include "gnome2-store/gkm-gnome2-file.h"

#include "gkm/gkm-crypto.h"

#include "egg/egg-secure-memory.h"

#include <glib.h>

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

EGG_SECURE_DEFINE_GLIB_GLOBALS ();

static void G_GNUC_NORETURN
failure (const gchar* message, ...)
{
	va_list va;
	va_start (va, message);
	vfprintf (stderr, message, va);
	fputc ('\n', stderr);
	va_end (va);
	exit (1);
}

int
main(int argc, char* argv[])
{
	const gchar *password;
	GkmDataResult res;
	GkmGnome2File *file;
	GkmSecret *login;
	int fd;

#if !GLIB_CHECK_VERSION(2,35,0)
	g_type_init ();
#endif
	gkm_crypto_initialize ();

	if (argc != 2)
		failure ("usage: dump-data-file filename");

	fd = open (argv[1], O_RDONLY, 0);
	if (fd == -1)
		failure ("dump-data-file: couldn't open file: %s: %s", argv[1], g_strerror (errno));

	password = getpass ("Password: ");
	login = gkm_secret_new ((guchar*)password, strlen (password));

	file = gkm_gnome2_file_new ();
	res = gkm_gnome2_file_read_fd (file, fd, login);
	g_object_unref (login);

	switch(res) {
	case GKM_DATA_FAILURE:
		failure ("dump-data-file: failed to read file: %s", argv[1]);
	case GKM_DATA_LOCKED:
		failure ("dump-data-file: invalid password for file: %s", argv[1]);
	case GKM_DATA_UNRECOGNIZED:
		failure ("dump-data-file: unparseable file format: %s", argv[1]);
	case GKM_DATA_SUCCESS:
		break;
	default:
		g_assert_not_reached ();
	}

	gkm_gnome2_file_dump (file);
	g_object_unref (file);

	return 0;
}
