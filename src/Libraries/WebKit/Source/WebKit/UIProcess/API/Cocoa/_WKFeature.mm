/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
#import "_WKFeatureInternal.h"
#import "_WKExperimentalFeature.h"
#import "_WKInternalDebugFeature.h"

#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKFeature

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKFeature.class, self))
        return;

    _wrappedFeature->API::Feature::~Feature();

    [super dealloc];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; name = %@; key = %@; defaultValue = %s >", NSStringFromClass(self.class), self, self.name, self.key, self.defaultValue ? "on" : "off"];
}

- (NSString *)name
{
    return _wrappedFeature->name();
}

- (WebFeatureStatus)status
{
    switch (_wrappedFeature->status()) {
    case API::FeatureStatus::Embedder:
        return WebFeatureStatusEmbedder;
    case API::FeatureStatus::Unstable:
        return WebFeatureStatusUnstable;
    case API::FeatureStatus::Internal:
        return WebFeatureStatusInternal;
    case API::FeatureStatus::Developer:
        return WebFeatureStatusDeveloper;
    case API::FeatureStatus::Testable:
        return WebFeatureStatusTestable;
    case API::FeatureStatus::Preview:
        return WebFeatureStatusPreview;
    case API::FeatureStatus::Stable:
        return WebFeatureStatusStable;
    case API::FeatureStatus::Mature:
        return WebFeatureStatusMature;
    default:
        ASSERT_NOT_REACHED();
    }
}

- (WebFeatureCategory)category
{
    switch (_wrappedFeature->category()) {
    case API::FeatureCategory::None:
        return WebFeatureCategoryNone;
    case API::FeatureCategory::Animation:
        return WebFeatureCategoryAnimation;
    case API::FeatureCategory::CSS:
        return WebFeatureCategoryCSS;
    case API::FeatureCategory::DOM:
        return WebFeatureCategoryDOM;
    case API::FeatureCategory::Extensions:
        return WebFeatureCategoryExtensions;
    case API::FeatureCategory::HTML:
        return WebFeatureCategoryHTML;
    case API::FeatureCategory::Javascript:
        return WebFeatureCategoryJavascript;
    case API::FeatureCategory::Media:
        return WebFeatureCategoryMedia;
    case API::FeatureCategory::Networking:
        return WebFeatureCategoryNetworking;
    case API::FeatureCategory::Privacy:
        return WebFeatureCategoryPrivacy;
    case API::FeatureCategory::Security:
        return WebFeatureCategorySecurity;
    default:
        ASSERT_NOT_REACHED();
    }
}

- (NSString *)key
{
    return _wrappedFeature->key();
}

- (NSString *)details
{
    return _wrappedFeature->details();
}

- (BOOL)defaultValue
{
    return _wrappedFeature->defaultValue();
}

- (BOOL)isHidden
{
    return _wrappedFeature->isHidden();
}

// For binary compatibility, some interfaces declare that they use the old
// _WKExperimentalFeature and _WKInternalDebugFeature classes, even though all
// instantiated features are actually instances of _WKFeature. Override
// isKindOfClass to prevent clients from detecting the change in instance type.
- (BOOL)isKindOfClass:(Class)aClass
{
    return [super isKindOfClass:aClass] || [aClass isEqual:[_WKExperimentalFeature class]] || [aClass isEqual:[_WKInternalDebugFeature class]];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_wrappedFeature;
}

@end
