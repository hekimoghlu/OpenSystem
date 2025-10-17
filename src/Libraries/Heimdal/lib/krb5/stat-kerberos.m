/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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
#include "krb5_locl.h"

#include <TargetConditionals.h>
#include <Foundation/Foundation.h>
#import <CoreAnalytics/CoreAnalytics.h>

void
_krb5_stat_ASREQ(krb5_context context, krb5_enctype userEnctype,
		 krb5_enctype asEnctype, const char *type, int fast)
{
    @autoreleasepool {
	AnalyticsSendEventLazy(@"com.apple.GSS.Kerberos", ^NSDictionary<NSString *,NSObject *> * _Nullable {
	    return @{
		@"AS_REQ_replykey_et" : @(asEnctype),
		@"AS_REQ_useret_et" : @(userEnctype),
		@"AS_REQ_preauth" : [NSString stringWithCString:type encoding:NSUTF8StringEncoding] ?: @"unknown",
		@"AS_REQ_FAST" : @(fast),
	    };
	});
    }
}
