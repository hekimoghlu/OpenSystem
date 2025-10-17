/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#ifndef _SKYWALK_NAMESPACE_PROTONS_H_
#define _SKYWALK_NAMESPACE_PROTONS_H_


/*
 * The protons module arbitrates IP protocol number usage across Skywalk and
 * the BSD networking stack. The IP protocol number is managed globally
 * regardless of interface or IP address.
 */

extern int protons_init(void);
extern void protons_fini(void);

/* opaque token representing a protocol namespace reservation. */
struct protons_token;

/*
 * Reserve a IP protocol number globally.
 * Reserved protocol namespace token is return via @ptp.
 */
extern int protons_reserve(struct protons_token **ptp, pid_t pid, pid_t epid,
    uint8_t proto);

/*
 * Release a IP protocol reservation recorded by the provided token.
 * *ptp will be reset to NULL after release.
 */
extern void protons_release(struct protons_token **ptp);

extern int protons_token_get_use_count(struct protons_token *pt);
extern bool protons_token_is_valid(struct protons_token *pt);
extern bool protons_token_has_matching_pid(struct protons_token *pt, pid_t pid,
    pid_t epid);

#endif /* !_SKYWALK_NAMESPACE_PROTONS_H_ */
