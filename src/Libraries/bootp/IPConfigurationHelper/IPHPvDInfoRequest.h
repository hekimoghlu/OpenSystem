/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#import <Foundation/Foundation.h>

#import "IPHPvDInfoRequestProtocol.h"

#pragma mark -
#pragma mark XPC

#define IPH_BUNDLE_ID_			"com.apple.IPConfiguration"
#define IPH_PRODUCT_NAME 		"IPConfigurationHelper"
#define IPH_BUNDLE_ID			IPH_BUNDLE_ID_ "." IPH_PRODUCT_NAME

#pragma mark -
#pragma mark PvD

#define kPvDInfoValidFetchXPCKey		CFSTR("valid_fetch")
#define kPvDInfoAdditionalInfoDictXPCKey	CFSTR("additional_information")

@interface IPHPvDInfoRequestServer : NSObject<IPHPvDInfoRequestProtocol, NSURLSessionTaskDelegate>
@end
