/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
//
// logging - message log support
//
#ifndef _H_LOGGING
#define _H_LOGGING

#include <security_utilities/utilities.h>
#include <sys/cdefs.h>

#ifdef _CPP_LOGGING
#pragma export on
#endif

namespace Security
{

//
// Log destination object
//
namespace Syslog
{

void syslog(int priority, const char *format, ...) __printflike(2, 3);

void emergency(const char *format, ...) __printflike(1, 2);
void alert(const char *format, ...) __printflike(1, 2);
void critical(const char *format, ...) __printflike(1, 2);
void error(const char *format, ...) __printflike(1, 2);
void warning(const char *format, ...) __printflike(1, 2);
void notice(const char *format, ...) __printflike(1, 2);
void info(const char *format, ...) __printflike(1, 2);
void debug(const char *format, ...) __printflike(1, 2);

void open(const char *ident, int facility, int options = 0);
	
int mask();
void upto(int priority);
void enable(int priority);
void disable(int priority);

} // end namespace Syslog

} // end namespace Security

#ifdef _CPP_LOGGING
#pragma export off
#endif

#endif //_H_LOGGING
