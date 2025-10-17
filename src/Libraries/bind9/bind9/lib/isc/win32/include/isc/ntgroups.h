/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
/* $Id: ntgroups.h,v 1.5 2007/06/19 23:47:20 tbox Exp $ */

#ifndef ISC_NTGROUPS_H
#define ISC_NTGROUPS_H 1

#include <isc/lang.h>
#include <isc/result.h>

ISC_LANG_BEGINDECLS


isc_result_t
isc_ntsecurity_getaccountgroups(char *name, char **Groups, unsigned int maxgroups,
	     unsigned int *total);

ISC_LANG_ENDDECLS

#endif /* ISC_NTGROUPS_H */
