/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
                 "recovery", recovery_key,
                 "[options]\n"
                 "    -R //generates and prints a string encoded key seed\n"
                 "    -G <string encoded seed> //Generate a recovery key from string encoded seed, but don't register\n"
                 "    -s <string encoded seed> //Set the recovery key\n"
                 "    -g                       //Get the recovery key\n"
                 "    -c                       //Clear the recovery key\n"
                 "    -V                       //Create Verifier Dictionary (printout)\n"
                 "    -F                       // prompt cdp to followup to repair the recovery key\n",
                 "Recovery Key Tool" )
