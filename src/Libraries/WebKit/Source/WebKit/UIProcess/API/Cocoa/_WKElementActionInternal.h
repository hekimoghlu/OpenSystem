/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#import "_WKElementAction.h"

#if PLATFORM(IOS_FAMILY)

@class WKActionSheetAssistant;
@class WKContentView;

UIActionIdentifier elementActionTypeToUIActionIdentifier(_WKElementActionType);

@interface _WKElementAction ()

+ (instancetype)_elementActionWithType:(_WKElementActionType)type info:(_WKActivatedElementInfo *)info assistant:(WKActionSheetAssistant *)assistant;
+ (instancetype)_elementActionWithType:(_WKElementActionType)type info:(_WKActivatedElementInfo *)info assistant:(WKActionSheetAssistant *)assistant disabled:(BOOL)disabled;
+ (instancetype)_elementActionWithType:(_WKElementActionType)type title:(NSString *)title actionHandler:(WKElementActionHandler)actionHandler;
- (void)_runActionWithElementInfo:(_WKActivatedElementInfo *)info forActionSheetAssistant:(WKActionSheetAssistant *)assistant;

@end

#endif // PLATFORM(IOS_FAMILY)
