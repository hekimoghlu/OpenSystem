/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
@class WebSecurityOrigin;

/*!
    @protocol WebQuotaManager
    @discussion This protocol is used to view and manipulate a per-origin storage quota.
*/
@protocol WebQuotaManager

/*!
    @method initWithOrigin:
    @param origin The security origin this will manage.
    @result A new WebQuotaManager object.
*/
- (id)initWithOrigin:(WebSecurityOrigin *)origin;

/*!
    @method origin
    @result The security origin this manager is managing.
*/
- (WebSecurityOrigin *)origin;

/*!
    @method usage
    @result The current total usage of all relevant items in this security origin in bytes.
*/
- (unsigned long long)usage;

/*!
    @method quota
    @result The current quota of security origin in bytes.
*/
- (unsigned long long)quota;

/*!
    @method setQuota:
    @param quota a new quota, in bytes, to set on this security origin.
*/
- (void)setQuota:(unsigned long long)quota;

@end
