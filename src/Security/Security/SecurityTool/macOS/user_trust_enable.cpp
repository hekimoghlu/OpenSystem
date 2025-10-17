/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#include "security_tool.h"
#include "user_trust_enable.h"
#include <errno.h>
#include <unistd.h>
#include <security_utilities/simpleprefs.h>
#include <Security/TrustSettingsSchema.h>		/* private SPI */
#include <CoreFoundation/CFNumber.h>

typedef enum {
	utoSet = 0,
	utoShow
} UserTrustOp;

int
user_trust_enable(int argc, char * const *argv)
{
	extern int optind;
	int arg;
	UserTrustOp op = utoShow;
	CFBooleanRef disabledBool = kCFBooleanFalse;	/* what we write to prefs */
	optind = 1;
	int ourRtn = 0;

	while ((arg = getopt(argc, argv, "deh")) != -1) {
		switch (arg) {
			case 'd':
				op = utoSet;
				disabledBool = kCFBooleanTrue;
				break;
			case 'e':
				op = utoSet;
				disabledBool = kCFBooleanFalse;
				break;
			default:
			case 'h':
				return SHOW_USAGE_MESSAGE;
		}
	}
	if(optind != argc) {
		return SHOW_USAGE_MESSAGE;
	}

	if(op == utoShow) {
		bool utDisable = false;

#if !defined MAC_OS_X_VERSION_10_6 || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_6
		Dictionary* prefsDict = new Dictionary(kSecTrustSettingsPrefsDomain, Dictionary::US_System);
#else
		Dictionary* prefsDict = Dictionary::CreateDictionary(kSecTrustSettingsPrefsDomain, Dictionary::US_System);
#endif
		if (prefsDict != NULL)
		{
			utDisable = prefsDict->getBoolValue(kSecTrustSettingsDisableUserTrustSettings);
			delete prefsDict;
		}

		fprintf(stdout, "User-level Trust Settings are %s\n",
			utDisable ? "Disabled" : "Enabled");
		return 0;
	}

	/*  set the pref... */
	if(geteuid() != 0) {
		fprintf(stderr, "You must be root to set this preference.\n");
		return 1;
	}

	/* get a mutable copy of the existing prefs, or a fresh empty one */
#if !defined MAC_OS_X_VERSION_10_6 || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_6
	MutableDictionary *prefsDict = new MutableDictionary(kSecTrustSettingsPrefsDomain, Dictionary::US_System);
#else
	MutableDictionary *prefsDict = MutableDictionary::CreateMutableDictionary(kSecTrustSettingsPrefsDomain, Dictionary::US_System);
#endif
	if (prefsDict == NULL)
	{
		prefsDict = new MutableDictionary();
	}

	prefsDict->setValue(kSecTrustSettingsDisableUserTrustSettings, disabledBool);
	if(prefsDict->writePlistToPrefs(kSecTrustSettingsPrefsDomain, Dictionary::US_System)) {
		fprintf(stdout, "...User-level Trust Settings are %s\n",
			(disabledBool == kCFBooleanTrue) ? "Disabled" : "Enabled");
	}
	else {
		fprintf(stderr, "Could not write system preferences.\n");
		ourRtn = 1;
	}
	delete prefsDict;
	return ourRtn;
}
