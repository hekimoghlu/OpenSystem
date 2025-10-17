/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
 * keychain_log.h
 * 
 * security tool subcommand (log) for sysdiagnose/ckcdiagnose information.
 *
 */


#include "SecurityTool/sharedTool/security_tool_commands.h"

SECURITY_COMMAND(
                 "synclog", keychain_log,
                 "[options]\n"
                 "    -s     sysdiagnose dump\n"
                 "    -i     info (current status)\n"
                 "    -D     [itemName]  dump contents of KVS\n"
                 "    -L     list all known view and their status\n"
                 "    -M string   place a mark in the syslog - category \"mark\"\n"
                 "\n",
                 "iCloud Keychain Logging")
