/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
 * Policy structure...
 */

typedef struct
{
  char			*name;		/* Policy name */
  cups_array_t		*job_access,	/* Private users/groups for jobs */
			*job_attrs,	/* Private attributes for jobs */
			*sub_access,	/* Private users/groups for subscriptions */
			*sub_attrs,	/* Private attributes for subscriptions */
			*ops;		/* Operations */
} cupsd_policy_t;

typedef struct cupsd_printer_s cupsd_printer_t;


/*
 * Globals...
 */

VAR cups_array_t	*Policies	VALUE(NULL);
					/* Policies */


/*
 * Prototypes...
 */

extern cupsd_policy_t	*cupsdAddPolicy(const char *policy);
extern cupsd_location_t	*cupsdAddPolicyOp(cupsd_policy_t *p,
			                  cupsd_location_t *po,
			                  ipp_op_t op);
extern http_status_t	cupsdCheckPolicy(cupsd_policy_t *p, cupsd_client_t *con,
				         const char *owner);
extern void		cupsdDeleteAllPolicies(void);
extern cupsd_policy_t	*cupsdFindPolicy(const char *policy);
extern cupsd_location_t	*cupsdFindPolicyOp(cupsd_policy_t *p, ipp_op_t op);
extern cups_array_t	*cupsdGetPrivateAttrs(cupsd_policy_t *p,
			                      cupsd_client_t *con,
					      cupsd_printer_t *printer,
			                      const char *owner);
