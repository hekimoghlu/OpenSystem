/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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
// Enables fake media support for PeerConnnectionFactory created from `deps` for
// testing purposes. Such fake media support ignores media dependencies in the
// `PeerConnectionFactoryDependencies`. Allows to test PeerConnection and
// PeerConnectionFactory in the presence of the media, but doesn't test media
// support itself.

#ifndef PC_TEST_ENABLE_FAKE_MEDIA_H_
#define PC_TEST_ENABLE_FAKE_MEDIA_H_

#include <memory>

#include "absl/base/nullability.h"
#include "api/peer_connection_interface.h"
#include "media/base/fake_media_engine.h"

namespace webrtc {

// Enables media support backed by the 'fake_media_engine'.
void EnableFakeMedia(
    PeerConnectionFactoryDependencies& deps,
    absl::Nonnull<std::unique_ptr<cricket::FakeMediaEngine>> fake_media_engine);

// Enables media support backed by unspecified lightweight fake implementation.
void EnableFakeMedia(PeerConnectionFactoryDependencies& deps);

}  // namespace webrtc

#endif  //  PC_TEST_ENABLE_FAKE_MEDIA_H_
