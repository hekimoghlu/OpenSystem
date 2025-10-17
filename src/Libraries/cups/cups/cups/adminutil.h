/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
#ifndef _CUPS_ADMINUTIL_H_
#  define _CUPS_ADMINUTIL_H_

/*
 * Include necessary headers...
 */

#  include <stdio.h>
#  include "cups.h"


/*
 * C++ magic...
 */

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */


/*
 * Constants...
 */

#  define CUPS_SERVER_DEBUG_LOGGING	"_debug_logging"
#  define CUPS_SERVER_REMOTE_ADMIN	"_remote_admin"
#  define CUPS_SERVER_REMOTE_ANY	"_remote_any"
#  define CUPS_SERVER_SHARE_PRINTERS	"_share_printers"
#  define CUPS_SERVER_USER_CANCEL_ANY	"_user_cancel_any"


/*
 * Types and structures...
 */

typedef void (*cups_device_cb_t)(const char *device_class,
                                 const char *device_id, const char *device_info,
                                 const char *device_make_and_model,
                                 const char *device_uri,
				 const char *device_location, void *user_data);
					/* Device callback
					 * @since CUPS 1.4/macOS 10.6@ */


/*
 * Functions...
 */

extern int	cupsAdminExportSamba(const char *dest, const char *ppd,
		                     const char *samba_server,
			             const char *samba_user,
				     const char *samba_password,
				     FILE *logfile) _CUPS_DEPRECATED;
extern char	*cupsAdminCreateWindowsPPD(http_t *http, const char *dest,
		                           char *buffer, int bufsize)
		                           _CUPS_DEPRECATED;

extern int	cupsAdminGetServerSettings(http_t *http,
			                   int *num_settings,
		                           cups_option_t **settings)
		                           _CUPS_API_1_3;
extern int	cupsAdminSetServerSettings(http_t *http,
		                           int num_settings,
		                           cups_option_t *settings)
		                           _CUPS_API_1_3;

extern ipp_status_t	cupsGetDevices(http_t *http, int timeout,
			               const char *include_schemes,
			               const char *exclude_schemes,
				       cups_device_cb_t callback,
				       void *user_data) _CUPS_DEPRECATED;


#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_CUPS_ADMINUTIL_H_ */
