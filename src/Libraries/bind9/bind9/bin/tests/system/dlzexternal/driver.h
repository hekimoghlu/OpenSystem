/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
/* $Id: driver.h,v 1.4 2011/03/17 09:25:54 fdupont Exp $ */

/*
 * This header includes the declarations of entry points.
 */

dlz_dlopen_version_t dlz_version;
dlz_dlopen_create_t dlz_create;
dlz_dlopen_destroy_t dlz_destroy;
dlz_dlopen_findzonedb_t dlz_findzonedb;
dlz_dlopen_lookup_t dlz_lookup;
dlz_dlopen_allowzonexfr_t dlz_allowzonexfr;
dlz_dlopen_allnodes_t dlz_allnodes;
dlz_dlopen_newversion_t dlz_newversion;
dlz_dlopen_closeversion_t dlz_closeversion;
dlz_dlopen_configure_t dlz_configure;
dlz_dlopen_ssumatch_t dlz_ssumatch;
dlz_dlopen_addrdataset_t dlz_addrdataset;
dlz_dlopen_subrdataset_t dlz_subrdataset;
dlz_dlopen_delrdataset_t dlz_delrdataset;
