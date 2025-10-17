/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
/* $Id$ */

#ifndef ISC_TASK_P_H
#define ISC_TASK_P_H

/*! \file */

#if defined(ISC_PLATFORM_USETHREADS)
void
isc__taskmgr_pause(isc_taskmgr_t *taskmgr);

void
isc__taskmgr_resume(isc_taskmgr_t *taskmgr);
#else
isc_boolean_t
isc__taskmgr_ready(isc_taskmgr_t *taskmgr);

isc_result_t
isc__taskmgr_dispatch(isc_taskmgr_t *taskmgr);
#endif /* !ISC_PLATFORM_USETHREADS */

#endif /* ISC_TASK_P_H */
