/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#include "SecurityTool/sharedTool/security_tool_commands.h"

SECURITY_COMMAND(
	"sync", keychain_sync,
	"[options]\n"
	"Keychain Syncing\n"
	"    -d     disable\n"
	"    -e     enable (join/create circle)\n"
	"    -i     info (current status)\n"
	"    -m     dump my peer\n"
	"\n"
	"Account/Circle Management\n"
	"    -a     accept all applicants\n"
	"    -r     reject all applicants\n"
    "    -b     device|all|single Register a backup bag - THIS RESETS BACKUPS!\n"
	"    -N     (re-)set to new account (USE WITH CARE: device will not leave circle before resetting account!)\n"
	"    -O     reset to offering\n"
	"    -R     reset circle\n"
    "    -o     list view unaware peers in circle\n"
    "    -0     boot view unaware peers from circle\n"
    "    -5     cleanup old KVS keys in KVS\n"
    "\n"
    "Circle Tools\n"
    "    --remove-peer SPID     Remove a peer identified by the first 8 or more\n"
    "                           characters of its spid. Specify multiple times to\n"
    "                           remove more than one peer.\n"
                 
    "    --enable-sos-compatibility    Enable SOS Compatibility Mode\n"
    "    --disable-sos-compatibility   Disable SOS Compatibility Mode\n"
    "    --fetch-sos-compatibility     Fetch SOS Compatibility Mode\n"
    "    --push-reset-circle           Push a SOS reset circle\n"
	"\n"
	"Password\n"
	"    -P     [label:]password  set password (optionally for a given label) for sync\n"
	"    -T     [label:]password  try password (optionally for a given label) for sync\n"
	"\n"
	"KVS\n"
	"    -k     pend all registered kvs keys\n"
	"    -C     clear all values from KVS\n"
	"    -D     [itemName]  dump contents of KVS\n"
	"    -W     sync and dump\n"
	"\n"
	"Misc\n"
	"    -v     [enable|disable|query:viewname] enable, disable, or query my PeerInfo's view set\n"
	"             viewnames are: keychain|masterkey|iclouddrive|photos|cloudkit|escrow|fde|maildrop|icloudbackup|notes|imessage|appletv|homekit\n"
    "                            wifi|passwords|creditcards|icloudidentity|othersyncable\n"
    "    -L     list all known view and their status\n"
	"    -U     purge private key material cache\n"
    "    -V     Report View Sync Status on all known clients.\n",
	"Keychain Syncing controls." )


