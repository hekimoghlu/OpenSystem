/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/*
 * IPMonitorControlPrefs.c
 * - definitions for accessing IPMonitor control preferences
 */

/*
 * Modification History
 *
 * January 14, 2013	Dieter Siegmund (dieter@apple)
 * - created
 */

#include "IPMonitorControlPrefs.h"

/*
 * kIPMonitorControlPrefsID
 * - identifies the IPMonitor preferences file that contains 'Verbose'
 */
#define kIPMonitorControlPrefsIDStr	"com.apple.IPMonitor.control.plist"

/*
 * kVerbose
 * - indicates whether IPMonitor is verbose or not
 */
#define kVerbose			CFSTR("Verbose")

/*
 * kDisableServiceCoupling
 * - indicates whether services should be coupled or not, by default
 *   DisableServiceCoupling is FALSE
 */
#define kDisableServiceCoupling		CFSTR("DisableServiceCoupling")

static _SCControlPrefsRef		S_control;

__private_extern__
_SCControlPrefsRef
IPMonitorControlPrefsInit(dispatch_queue_t queue,
			  IPMonitorControlPrefsCallBack	callback)
{
	return _SCControlPrefsCreateWithQueue(kIPMonitorControlPrefsIDStr,
					      queue, callback);
}

/*
 * Verbose
 */
__private_extern__ Boolean
IPMonitorControlPrefsIsVerbose(void)
{
	Boolean	verbose	= FALSE;

	if (S_control == NULL) {
		S_control = IPMonitorControlPrefsInit(NULL, NULL);
	}
	if (S_control != NULL) {
		verbose = _SCControlPrefsGetBoolean(S_control, kVerbose);
	}
	return verbose;
}

__private_extern__ Boolean
IPMonitorControlPrefsSetVerbose(Boolean verbose)
{
	Boolean	ok	= FALSE;

	if (S_control == NULL) {
		S_control = IPMonitorControlPrefsInit(NULL, NULL);
	}
	if (S_control != NULL) {
		ok = _SCControlPrefsSetBoolean(S_control, kVerbose, verbose);
	}
	return ok;
}

/*
 * DisableServiceCoupling
 */
__private_extern__ Boolean
IPMonitorControlPrefsGetDisableServiceCoupling(void)
{
	Boolean	disable_coupling = FALSE;

	if (S_control == NULL) {
		S_control = IPMonitorControlPrefsInit(NULL, NULL);
	}
	if (S_control != NULL) {
		disable_coupling = _SCControlPrefsGetBoolean(S_control,
						     kDisableServiceCoupling);
	}
	return disable_coupling;
}
__private_extern__ Boolean
IPMonitorControlPrefsSetDisableServiceCoupling(Boolean disable_coupling)
{
	Boolean	ok	= FALSE;

	if (S_control == NULL) {
		S_control = IPMonitorControlPrefsInit(NULL, NULL);
	}
	if (S_control != NULL) {
		ok = _SCControlPrefsSetBoolean(S_control,
					       kDisableServiceCoupling,
					       disable_coupling);
	}
	return ok;
}

#ifdef TEST_IPMONITORCONTROLPREFS

static void
usage(const char * progname)
{
	fprintf(stderr,
		"usage:\n"
		"\t%s set (verbose | disable-service-coupling) ( on | off )\n"
		"\t%s get (verbose | disable-service-coupling)\n",
		progname, progname);
	exit(1);
}

int
main(int argc, char * argv[])
{
	const char *	command;
	Boolean		do_set = FALSE;
	Boolean		do_verbose = FALSE;
	const char *	progname;
	const char *	type;

	progname = argv[0];
	if (argc < 3) {
		usage(progname);
	}
	command = argv[1];
	if (strcasecmp(command, "set") == 0) {
		do_set = TRUE;
	}
	else if (strcasecmp(command, "get") == 0) {
		do_set = FALSE;
	}
	else {
		usage(progname);
	}
	type = argv[2];
	if (strcasecmp(type, "verbose") == 0) {
		do_verbose = TRUE;
	}
	else if (strcasecmp(type, "disable-service-coupling") == 0) {
		do_verbose = FALSE;
	}
	else {
		usage(progname);
	}
	if (do_set) {
		const char *	on_off;
		Boolean		val = FALSE;
		Boolean		success;

		if (argc < 4) {
			usage(progname);
		}
		on_off = argv[3];
		if (strcasecmp(on_off, "on") == 0) {
			val = TRUE;
		}
		else if (strcasecmp(on_off, "off") == 0) {
			val = FALSE;
		}
		else {
			usage(progname);
		}
		if (do_verbose) {
			success = IPMonitorControlPrefsSetVerbose(val);
		}
		else {
			success = IPMonitorControlPrefsSetDisableServiceCoupling(val);
		}
		if (!success) {
			fprintf(stderr, "failed to save prefs\n");
			exit(2);
		}
	}
	else {
		Boolean	val;

		if (do_verbose) {
			val = IPMonitorControlPrefsIsVerbose();
			printf("Verbose is %s\n",
			       val ? "true" : "false");
		}
		else {
			val = IPMonitorControlPrefsGetDisableServiceCoupling();
			printf("disable-service-coupling is %s\n",
			       val ? "true" : "false");
		}
	}
	exit(0);
	return (0);
}

#endif /* TEST_IPMONITORCONTROLPREFS */
