/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
#include <TargetConditionals.h>

#define SHOW_USAGE_MESSAGE 2

// This is included to make SECURITY_COMMAND macros result in declarations of
// commands for use in SecurityTool

#ifndef SECURITY_COMMAND
#define SECURITY_COMMAND(name, function, parameters, description) int function(int argc, char * const *argv);

#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
#define SECURITY_COMMAND_IOS(name, function, parameters, description) int function(int argc, char * const *argv);
#else
#define SECURITY_COMMAND_IOS(name, function, parameters, description) extern int command_not_on_this_platform(int argc, char * const *argv);
#endif

#if !TARGET_OS_IPHONE
#define SECURITY_COMMAND_MAC(name, function, parameters, description) int function(int argc, char * const *argv);
#else
#define SECURITY_COMMAND_MAC(name, function, parameters, description) extern int command_not_on_this_platform(int argc, char * const *argv);
#endif

#endif
