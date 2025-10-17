/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
 * Modification History
 *
 * May 29, 2003			Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _PREFS_H
#define _PREFS_H

#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <SystemConfiguration/SystemConfiguration.h>


extern Boolean	_prefs_changed;


__BEGIN_DECLS

#define DISABLE_UNTIL_NEEDED	"disable-until-needed"
#define DISABLE_PRIVATE_RELAY	"disable-private-relay"
#define ENABLE_LOW_DATA_MODE	"enable-low-data-mode"
#define OVERRIDE_EXPENSIVE	"override-expensive"
#define DISABLE_SERVICE_COUPLING "disable-service-coupling"

#define ALLOW_NEW_INTERFACES	"allow-new-interfaces"
#define CONFIGURE_NEW_INTERFACES "configure-new-interfaces"

AuthorizationRef	_prefs_AuthorizationCreate	(void);
void			_prefs_AuthorizationFree	(AuthorizationRef authorization);

Boolean	_prefs_open		(CFStringRef name, const char *path);
void	_prefs_save		(void);
void	_prefs_close		(void);
Boolean	_prefs_commitRequired	(int argc, char * const argv[], const char *command);

int	findPref		(char *pref);
void	do_getPref		(char *pref, int argc, char * const argv[]);
void	do_setPref		(char *pref, int argc, char * const argv[]);

void	do_prefs_init		(void);
void	do_prefs_quit		(int argc, char * const argv[]);

void	do_prefs_open		(int argc, char * const argv[]);
void	do_prefs_lock		(int argc, char * const argv[]);
void	do_prefs_unlock		(int argc, char * const argv[]);
void	do_prefs_commit		(int argc, char * const argv[]);
void	do_prefs_apply		(int argc, char * const argv[]);
void	do_prefs_close		(int argc, char * const argv[]);
void	do_prefs_synchronize	(int argc, char * const argv[]);

void	do_prefs_list		(int argc, char * const argv[]);
void	do_prefs_get		(int argc, char * const argv[]);
void	do_prefs_set		(int argc, char * const argv[]);
void	do_prefs_remove		(int argc, char * const argv[]);

void	do_log			(char *pref, int argc, char * const argv[]);
void	do_disable_service_coupling	(int argc, char * const argv[]);
void	do_disable_until_needed	(int argc, char * const argv[]);
void	do_disable_private_relay(int argc, char * const argv[]);
void	do_enable_low_data_mode	(int argc, char * const argv[]);
void	do_override_expensive	(int argc, char * const argv[]);


#if	!TARGET_OS_IPHONE
void	do_allow_new_interfaces	(char *pref, int argc, char * const argv[]);
#endif	// !TARGET_OS_IPHONE
void	do_configure_new_interfaces(char *pref, int argc, char * const argv[]);

__END_DECLS

#endif /* !_PREFS_H */
