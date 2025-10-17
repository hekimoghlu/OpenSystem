/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
// This file contains the WebRTC suppressions for LeakSanitizer.
// You can also pass additional suppressions via LSAN_OPTIONS:
// LSAN_OPTIONS=suppressions=/path/to/suppressions. Please refer to
// http://dev.chromium.org/developers/testing/leaksanitizer for more info.

#if defined(LEAK_SANITIZER)

// Please make sure the code below declares a single string variable
// kLSanDefaultSuppressions which contains LSan suppressions delimited by
// newlines. See http://dev.chromium.org/developers/testing/leaksanitizer
// for the instructions on writing suppressions.
char kLSanDefaultSuppressions[] =

    // ============ Leaks in third-party code shared with Chromium =============
    // These entries are copied from build/sanitizers/lsan_suppressions.cc in
    // Chromium. Please don't add new entries here unless they're present in
    // there.

    // False positives in libfontconfig. http://crbug.com/39050
    "leak:libfontconfig\n"

    // Leaks in Nvidia's libGL.
    "leak:libGL.so\n"

    // XRandR has several one time leaks.
    "leak:libxrandr\n"

    // xrandr leak. http://crbug.com/119677
    "leak:XRRFindDisplay\n"

    // ========== Leaks in third-party code not shared with Chromium ===========

    // None known so far.

    // ================ Leaks in WebRTC code ================
    // PLEASE DO NOT ADD SUPPRESSIONS FOR NEW LEAKS.
    // Instead, commits that introduce memory leaks should be reverted.
    // Suppressing the leak is acceptable in some cases when reverting is
    // impossible, i.e. when enabling leak detection for the first time for a
    // test target with pre-existing leaks.

    // rtc_unittest
    // https://code.google.com/p/webrtc/issues/detail?id=3827 for details.
    "leak:rtc::unstarted_task_test_DoNotDeleteTask2_Test::TestBody\n"
    "leak:rtc::HttpServer::HandleConnection\n"
    "leak:rtc::HttpServer::Connection::onHttpHeaderComplete\n"
    "leak:rtc::HttpResponseData::set_success\n"
    "leak:rtc::HttpData::changeHeader\n"
    // https://code.google.com/p/webrtc/issues/detail?id=4149 for details.
    "leak:StartDNSLookup\n"

    // rtc_media_unittests
    "leak:cricket::FakeNetworkInterface::SetOption\n"
    "leak:CodecTest_TestCodecOperators_Test::TestBody\n"
    "leak:VideoEngineTest*::ConstrainNewCodecBody\n"
    "leak:VideoMediaChannelTest*::AddRemoveRecvStreams\n"
    "leak:WebRtcVideoCapturerTest_TestCapture_Test::TestBody\n"
    "leak:WebRtcVideoEngineTestFake_MultipleSendStreamsWithOneCapturer_Test::"
    "TestBody\n"
    "leak:WebRtcVideoEngineTestFake_SetBandwidthInConference_Test::TestBody\n"
    "leak:WebRtcVideoEngineTestFake_SetSendCodecsRejectBadFormat_Test::"
    "TestBody\n"

    // peerconnection_unittests
    // https://code.google.com/p/webrtc/issues/detail?id=2528
    "leak:cricket::FakeVideoMediaChannel::~FakeVideoMediaChannel\n"
    "leak:DtmfSenderTest_InsertEmptyTonesToCancelPreviousTask_Test::TestBody\n"
    "leak:sigslot::_signal_base2*::~_signal_base2\n"
    "leak:testing::internal::CmpHelperEQ\n"
    "leak:webrtc::AudioDeviceLinuxALSA::InitMicrophone\n"
    "leak:webrtc::AudioDeviceLinuxALSA::InitSpeaker\n"
    "leak:webrtc::CreateIceCandidate\n"
    "leak:webrtc::WebRtcIdentityRequestObserver::OnSuccess\n"
    "leak:PeerConnectionInterfaceTest_SsrcInOfferAnswer_Test::TestBody\n"
    "leak:PeerConnectionInterfaceTest_CloseAndTestMethods_Test::TestBody\n"
    "leak:WebRtcSdpTest::TestDeserializeRtcpFb\n"
    "leak:WebRtcSdpTest::TestSerialize\n"
    "leak:WebRtcSdpTest_SerializeSessionDescriptionWithDataChannelAndBandwidth_"
    "Test::TestBody\n"
    "leak:WebRtcSdpTest_SerializeSessionDescriptionWithBandwidth_Test::"
    "TestBody\n"
    "leak:WebRtcSessionTest::SetLocalDescriptionExpectError\n"
    "leak:WebRtcSessionTest_TestAVOfferWithAudioOnlyAnswer_Test::TestBody\n"

    // PLEASE READ ABOVE BEFORE ADDING NEW SUPPRESSIONS.

    // End of suppressions.
    ;  // Please keep this semicolon.

#endif  // LEAK_SANITIZER
