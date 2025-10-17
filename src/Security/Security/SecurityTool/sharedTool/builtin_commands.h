/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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
#include <TargetConditionals.h>

SECURITY_COMMAND("help", help,
                 "[command ...]",
                 "Show all commands. Or show usage for a command.")

SECURITY_COMMAND("digest", command_digest,
                 "algo file(s)...\n"
                 "Where algo is one of:\n"
                 "    sha1\n"
                 "    sha256\n"
                 "    sha512\n",
                 "Calculate a digest over the given file(s).")

SECURITY_COMMAND("whoami", command_whoami,
                 "",
                 "Ask securityd who you are.")

#if !TARGET_OS_BRIDGE
SECURITY_COMMAND("sos-stats", command_sos_stats,
                 "",
                 "SOS for performance numbers.")

SECURITY_COMMAND("sos-control", command_sos_control,
                 "",
                 "SOS control.")

SECURITY_COMMAND("bubble", command_bubble,
                 "",
                 "Transfer to sync bubble")

SECURITY_COMMAND("system-transfer", command_system_transfer,
                 "",
                 "Transfer (transmogrify) items to system keychain")

SECURITY_COMMAND("system-transcrypt", command_system_transcrypt,
                 "",
                 "Transcrypt emulated system keychain items to system keychain keybag")

SECURITY_COMMAND("watchdog", command_watchdog,
                     "[parameter ...]\n"
                     "Where parameter is one of:\n"
                     "    allowed-runtime <x>\n"
                     "    reset-period <x>\n"
                     "    check-period <x>\n"
                     "    graceful-exit-time <x>\n",
                     "Show current watchdog parameters or set an individual parameter")

SECURITY_COMMAND("sos-bypass", command_bypass,
                 "\n  s: set bypass bit\n"
                 "  c: clear bypass bit\n",
                 "Setup SOSAccount to bypass checks and grow the circle for account stuffing testing.")
#endif

SECURITY_COMMAND("keychain-check", command_keychain_check,
                    "",
                    "check the status of your keychain to determine if there are any items we can't decrypt")

SECURITY_COMMAND("keychain-cleanup", command_keychain_cleanup,
                    "",
                    "attempt to remove keychain items we can no longer decrypt")
