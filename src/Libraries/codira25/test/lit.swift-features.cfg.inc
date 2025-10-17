/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
 * This source file is part of the Codira.org open source project
 *
 * Copyright (c) 2014 - 2025 Apple Inc. and the Codira project authors
 * Licensed under Apache License v2.0 with Runtime Library Exception
 *
 * See https://language.org/LICENSE.txt for license information
 * See https://language.org/CONTRIBUTORS.txt for the list of Codira project authors
 */

/*
 * This is a configuration file for the 'lit' test runner. It takes the
 * Features.def for the language and creates 'lit' features associated with
 * them when the Codira feature is available. It allows using
 * `REQUIRES: FeatureName` instead of `REQUIRES: asserts`, which tend to be
 * left behind when the Codira feature becomes non-experimental.
 *
 * This file is preprocessed the Clang preprocessor and generates Python files.
 * C comments are possible, but not Python comments (they look like preprocessor
 * statements).
 */

config.existing_language_features = set()
def language_feature(feature_name, enabled):
    config.existing_language_features.add(feature_name)
    if enabled or "asserts" in config.available_features:
        config.available_features.add("language_feature_" + feature_name)

#define UPCOMING_FEATURE(FeatureName, SENumber, Version) language_feature(#FeatureName, True)
#define EXPERIMENTAL_FEATURE(FeatureName, AvailableInProd) language_feature(#FeatureName, #AvailableInProd == "true")
#define LANGUAGE_FEATURE(FeatureName, SENumber, Description)                   \
  language_feature(#FeatureName, True)

#include <language/Basic/Features.def>
