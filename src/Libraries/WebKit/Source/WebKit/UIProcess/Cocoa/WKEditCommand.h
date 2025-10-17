/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#import <wtf/RefPtr.h>

namespace WebKit {
class WebEditCommandProxy;
}

@interface WKEditCommand : NSObject {
@private
    RefPtr<WebKit::WebEditCommandProxy> _command;
}
- (instancetype)initWithWebEditCommandProxy:(Ref<WebKit::WebEditCommandProxy>&&)command;
- (instancetype)init NS_UNAVAILABLE;
- (WebKit::WebEditCommandProxy&)command;
@end

// WKEditorUndoTarget serves as the target when registering with the platform undo manager,
// and is only used in conjunction with WKEditCommand.
@interface WKEditorUndoTarget : NSObject
- (void)undoEditing:(id)sender;
- (void)redoEditing:(id)sender;
@end
