/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
/* Current release of ppp, MUST be changed before submission */
#define CURRENT_RELEASE_TAG		"884"

/* Current working tag */
#define CURRENT_DEVELOPMENT_TAG		"3468584"


#if (!defined(DEVELOPMENT))

/* Development version of ppp */
#define PPP_VERSION		CURRENT_RELEASE_TAG " [Engineering build " CURRENT_DEVELOPMENT_TAG ", " __DATE__ " " __TIME__ "]"

#else

/* Release version pf ppp */
#define PPP_VERSION		CURRENT_RELEASE_TAG

#endif
