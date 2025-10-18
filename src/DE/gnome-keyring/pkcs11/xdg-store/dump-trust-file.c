/*
 * gnome-keyring
 *
 * Copyright 2010 (C) Collabora Ltd.
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
 *
 * Author: Stef Walter <stefw@collabora.co.uk>
 */

#include "config.h"

#include "egg/egg-asn1x.h"
#include "egg/egg-asn1-defs.h"
#include "egg/egg-dn.h"
#include "egg/egg-error.h"
#include "egg/egg-hex.h"

#include <stdlib.h>

/* Bring in the relevant definitions */
#include "xdg-store/gkm-xdg-asn1-defs.h"

static void
barf_and_die (const char *msg, const char *detail)
{
	if (detail)
		g_printerr ("dump-trust-file: %s: %s\n", msg, detail);
	else
		g_printerr ("dump-trust-file: %s\n", msg);
	exit (1);
}

static void
dump_certificate_reference (GNode *asn)
{
	gchar *issuer, *serial;
	GBytes *data;
	GNode *name;
	GBytes *element;

	/* Parse the name out */
	name = egg_asn1x_create (pkix_asn1_tab, "Name");
	g_return_if_fail (name);
	element = egg_asn1x_get_element_raw (egg_asn1x_node (asn, "issuer", NULL));
	g_return_if_fail (element);
	if (!egg_asn1x_decode (name, element))
		barf_and_die ("couldn't parse certificate", egg_asn1x_message (name));
	g_bytes_unref (element);

	issuer = egg_dn_read (egg_asn1x_node (name, "rdnSequence", NULL));
	g_return_if_fail (issuer);

	data = egg_asn1x_get_integer_as_raw (egg_asn1x_node (asn, "serialNumber", NULL));
	g_return_if_fail (data != NULL);
	serial = egg_hex_encode (g_bytes_get_data (data, NULL), g_bytes_get_size (data));

	g_print ("Reference\n");
	g_print ("    issuer: %s\n", issuer);
	g_print ("    serial: 0x%s\n", serial);

	egg_asn1x_destroy (name);

	g_free (serial);
	g_free (issuer);
}

static void
dump_certificate_complete (GNode *asn)
{
	GNode *cert;
	gchar *issuer, *serial, *subject;
	GBytes *element;
	GBytes *data;

	/* Parse the certificate out */
	cert = egg_asn1x_create (pkix_asn1_tab, "Certificate");
	g_return_if_fail (cert);

	element = egg_asn1x_get_element_raw (asn);
	g_return_if_fail (element);
	if (!egg_asn1x_decode (cert, element))
		barf_and_die ("couldn't parse certificate", egg_asn1x_message (cert));
	g_bytes_unref (element);

	issuer = egg_dn_read (egg_asn1x_node (cert, "tbsCertificate", "issuer", "rdnSequence", NULL));
	g_return_if_fail (issuer);

	subject = egg_dn_read (egg_asn1x_node (cert, "tbsCertificate", "subject", "rdnSequence", NULL));
	g_return_if_fail (subject);

	data = egg_asn1x_get_integer_as_raw (egg_asn1x_node (cert, "tbsCertificate", "serialNumber", NULL));
	g_return_if_fail (data != NULL);
	serial = egg_hex_encode (g_bytes_get_data (data, NULL), g_bytes_get_size (data));
	g_bytes_unref (data);

	g_print ("Complete\n");
	g_print ("    issuer: %s\n", issuer);
	g_print ("    subject: %s\n", subject);
	g_print ("    serial: 0x%s\n", serial);

	egg_asn1x_destroy (cert);

	g_free (serial);
	g_free (issuer);
	g_free (subject);
}


static void
dump_assertion (GNode *asn)
{
	gchar *purpose, *peer;
	GQuark level;

	purpose = egg_asn1x_get_string_as_utf8 (egg_asn1x_node (asn, "purpose", NULL), NULL);
	g_return_if_fail (purpose);

	level = egg_asn1x_get_enumerated (egg_asn1x_node (asn, "level", NULL));
	g_return_if_fail (level);

	if (egg_asn1x_have (egg_asn1x_node (asn, "peer", NULL)))
		peer = egg_asn1x_get_string_as_utf8 (egg_asn1x_node (asn, "peer", NULL), NULL);
	else
		peer = NULL;

	g_print ("Assertion\n");
	g_print ("    purpose: %s\n", purpose);
	g_print ("    level: %s\n", g_quark_to_string (level));
	if (peer)
		g_print ("    peer: %s\n", peer);

	g_free (purpose);
	g_free (peer);
}

int
main(int argc, char* argv[])
{
	GError *err = NULL;
	gchar *contents;
	gsize n_contents;
	GNode *asn, *node;
	GBytes *bytes;
	gint i, count;

	if (argc != 2) {
		g_printerr ("usage: dump-trust-file file\n");
		return 2;
	}

	if (!g_file_get_contents (argv[1], &contents, &n_contents, &err))
		barf_and_die ("couldn't load file", egg_error_message (err));

	asn = egg_asn1x_create (xdg_asn1_tab, "trust-1");
	g_return_val_if_fail (asn, 1);

	bytes = g_bytes_new_take (contents, n_contents);
	if (!egg_asn1x_decode (asn, bytes))
		barf_and_die ("couldn't parse file", egg_asn1x_message (asn));
	g_bytes_unref (bytes);

	/* Print out the certificate we refer to first */
	node = egg_asn1x_node (asn, "reference", "certReference", NULL);
	if (egg_asn1x_have (node)) {
		dump_certificate_reference (node);
	} else {
		node = egg_asn1x_node (asn, "reference", "certComplete", NULL);
		if (egg_asn1x_have (node))
			dump_certificate_complete (node);
		else
			barf_and_die ("unsupported certificate reference", NULL);
	}

	/* Then the assertions */
	count = egg_asn1x_count (egg_asn1x_node (asn, "assertions", NULL));
	for (i = 0; i < count; ++i) {
		node = egg_asn1x_node (asn, "assertions", i + 1, NULL);
		dump_assertion (node);
	}

	egg_asn1x_destroy (asn);

	return 0;
}
