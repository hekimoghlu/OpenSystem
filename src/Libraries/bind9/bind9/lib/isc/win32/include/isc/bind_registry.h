/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
/* $Id: bind_registry.h,v 1.8 2007/06/19 23:47:20 tbox Exp $ */

#ifndef ISC_BINDREGISTRY_H
#define ISC_BINDREGISTRY_H

/*
 * BIND makes use of the following Registry keys in various places, especially
 * during startup and installation
 */

#define BIND_SUBKEY		"Software\\ISC\\BIND"
#define BIND_SESSION		"CurrentSession"
#define BIND_SESSION_SUBKEY	"Software\\ISC\\BIND\\CurrentSession"
#define BIND_UNINSTALL_SUBKEY	\
	"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\ISC BIND"

#define EVENTLOG_APP_SUBKEY	\
	"SYSTEM\\CurrentControlSet\\Services\\EventLog\\Application"
#define BIND_MESSAGE_SUBKEY	\
	"SYSTEM\\CurrentControlSet\\Services\\EventLog\\Application\\named"
#define BIND_MESSAGE_NAME	"named"

#define BIND_SERVICE_SUBKEY	\
	"SYSTEM\\CurrentControlSet\\Services\\named"


#define BIND_CONFIGFILE		0
#define BIND_DEBUGLEVEL		1
#define BIND_QUERYLOG		2
#define BIND_FOREGROUND		3
#define BIND_PORT		4

#endif /* ISC_BINDREGISTRY_H */
