/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef _CUPS_BACKEND_H_
#  define _CUPS_BACKEND_H_


/*
 * Include necessary headers...
 */

#  include "versioning.h"


/*
 * C++ magic...
 */

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */

/*
 * Constants...
 */

enum cups_backend_e			/**** Backend exit codes ****/
{
  CUPS_BACKEND_OK = 0,			/* Job completed successfully */
  CUPS_BACKEND_FAILED = 1,		/* Job failed, use error-policy */
  CUPS_BACKEND_AUTH_REQUIRED = 2,	/* Job failed, authentication required */
  CUPS_BACKEND_HOLD = 3,		/* Job failed, hold job */
  CUPS_BACKEND_STOP = 4,		/* Job failed, stop queue */
  CUPS_BACKEND_CANCEL = 5,		/* Job failed, cancel job */
  CUPS_BACKEND_RETRY = 6,		/* Job failed, retry this job later */
  CUPS_BACKEND_RETRY_CURRENT = 7	/* Job failed, retry this job immediately */
};
typedef enum cups_backend_e cups_backend_t;
					/**** Backend exit codes ****/


/*
 * Prototypes...
 */

extern const char	*cupsBackendDeviceURI(char **argv) _CUPS_API_1_2;
extern void		cupsBackendReport(const char *device_scheme,
			                  const char *device_uri,
			                  const char *device_make_and_model,
			                  const char *device_info,
			                  const char *device_id,
			                  const char *device_location)
					  _CUPS_API_1_4;


#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_CUPS_BACKEND_H_ */
