/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
#import "config.h"
#import "WKPageHostedModelView.h"

#if PLATFORM(IOS_FAMILY) && ENABLE(MODEL_PROCESS)

#import <wtf/RetainPtr.h>

@implementation WKPageHostedModelView {
    RetainPtr<UIView> _remoteModelView;
}

- (UIView *)remoteModelView
{
    return _remoteModelView.get();
}

- (void)setRemoteModelView:(UIView *)remoteModelView
{
    if (_remoteModelView.get() == remoteModelView)
        return;

    [_remoteModelView removeFromSuperview];

    _remoteModelView = remoteModelView;
    CGRect bounds = self.bounds;
    [_remoteModelView setFrame:bounds];
    [_remoteModelView setAutoresizingMask:(UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight)];
    [self addSubview:_remoteModelView.get()];
}

@end

#endif // PLATFORM(IOS_FAMILY) && ENABLE(MODEL_PROCESS)
