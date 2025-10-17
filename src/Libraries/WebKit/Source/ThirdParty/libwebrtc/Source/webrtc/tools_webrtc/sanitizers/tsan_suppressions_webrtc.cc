/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
// This file contains the WebRTC suppressions for ThreadSanitizer.
// Please refer to
// http://dev.chromium.org/developers/testing/threadsanitizer-tsan-v2
// for more info.

#if defined(THREAD_SANITIZER)

// Please make sure the code below declares a single string variable
// kTSanDefaultSuppressions contains TSan suppressions delimited by newlines.
// See http://dev.chromium.org/developers/testing/threadsanitizer-tsan-v2
// for the instructions on writing suppressions.
char kTSanDefaultSuppressions[] =

    // WebRTC specific suppressions.

    // Split up suppressions covered previously by thread.cc and
    // messagequeue.cc.
    "race:vp8cx_remove_encoder_threads\n"
    "race:third_party/libvpx/source/libvpx/vp9/common/vp9_scan.h\n"

    // rtc_unittests
    // https://code.google.com/p/webrtc/issues/detail?id=2080
    "race:rtc_base/logging.cc\n"

    // Potential deadlocks detected after roll in r6516.
    // https://code.google.com/p/webrtc/issues/detail?id=3509
    "deadlock:webrtc::test::UdpSocketManagerPosixImpl::RemoveSocket\n"

    // TODO(pbos): Trace events are racy due to lack of proper POD atomics.
    // https://code.google.com/p/webrtc/issues/detail?id=2497
    "race:*trace_event_unique_catstatic*\n"

    // Race between InitCpuFlags and TestCpuFlag in libyuv.
    // https://code.google.com/p/libyuv/issues/detail?id=508
    "race:InitCpuFlags\n"

    // Test-only race due to PeerConnection::session() being virtual for
    // testing. The stats collector may call session() before or after the
    // destructor begins executing, which modifies the vtable.
    "race:*RTCStatsIntegrationTest_GetsStatsWhileDestroyingPeerConnections_"
    "Test::TestBody\n"

    // http://crbug.com/244856
    "race:libpulsecommon*.so\n"

    // https://crbug.com/1158622
    "race:absl::synchronization_internal::Waiter::Post\n"

    // End of suppressions.
    ;  // Please keep this semicolon.

#endif  // THREAD_SANITIZER
