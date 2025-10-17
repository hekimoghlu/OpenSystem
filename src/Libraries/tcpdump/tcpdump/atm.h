/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
 * Traffic types for ATM.
 */
#define ATM_UNKNOWN	0	/* Unknown */
#define ATM_LANE	1	/* LANE */
#define ATM_LLC		2	/* LLC encapsulation */

/*
 * some OAM cell captures (most notably Juniper's)
 * do not deliver a heading HEC byte
 */
#define ATM_OAM_NOHEC   0
#define ATM_OAM_HEC     1
#define ATM_HDR_LEN_NOHEC 4
