/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
#ifndef SDK_ANDROID_SRC_JNI_PC_PEER_CONNECTION_H_
#define SDK_ANDROID_SRC_JNI_PC_PEER_CONNECTION_H_

#include <map>
#include <memory>
#include <vector>

#include "api/peer_connection_interface.h"
#include "pc/media_stream_observer.h"
#include "sdk/android/src/jni/jni_helpers.h"
#include "sdk/android/src/jni/pc/media_constraints.h"
#include "sdk/android/src/jni/pc/media_stream.h"
#include "sdk/android/src/jni/pc/rtp_receiver.h"
#include "sdk/android/src/jni/pc/rtp_transceiver.h"

namespace webrtc {
namespace jni {

void JavaToNativeRTCConfiguration(
    JNIEnv* jni,
    const JavaRef<jobject>& j_rtc_config,
    PeerConnectionInterface::RTCConfiguration* rtc_config);

rtc::KeyType GetRtcConfigKeyType(JNIEnv* env,
                                 const JavaRef<jobject>& j_rtc_config);

ScopedJavaLocalRef<jobject> NativeToJavaAdapterType(JNIEnv* env,
                                                    int adapterType);

// Adapter between the C++ PeerConnectionObserver interface and the Java
// PeerConnection.Observer interface.  Wraps an instance of the Java interface
// and dispatches C++ callbacks to Java.
class PeerConnectionObserverJni : public PeerConnectionObserver {
 public:
  PeerConnectionObserverJni(JNIEnv* jni, const JavaRef<jobject>& j_observer);
  ~PeerConnectionObserverJni() override;

  // Implementation of PeerConnectionObserver interface, which propagates
  // the callbacks to the Java observer.
  void OnIceCandidate(const IceCandidateInterface* candidate) override;
  void OnIceCandidateError(const std::string& address,
                           int port,
                           const std::string& url,
                           int error_code,
                           const std::string& error_text) override;

  void OnIceCandidatesRemoved(
      const std::vector<cricket::Candidate>& candidates) override;
  void OnSignalingChange(
      PeerConnectionInterface::SignalingState new_state) override;
  void OnIceConnectionChange(
      PeerConnectionInterface::IceConnectionState new_state) override;
  void OnStandardizedIceConnectionChange(
      PeerConnectionInterface::IceConnectionState new_state) override;
  void OnConnectionChange(
      PeerConnectionInterface::PeerConnectionState new_state) override;
  void OnIceConnectionReceivingChange(bool receiving) override;
  void OnIceGatheringChange(
      PeerConnectionInterface::IceGatheringState new_state) override;
  void OnIceSelectedCandidatePairChanged(
      const cricket::CandidatePairChangeEvent& event) override;
  void OnAddStream(rtc::scoped_refptr<MediaStreamInterface> stream) override;
  void OnRemoveStream(rtc::scoped_refptr<MediaStreamInterface> stream) override;
  void OnDataChannel(rtc::scoped_refptr<DataChannelInterface> channel) override;
  void OnRenegotiationNeeded() override;
  void OnAddTrack(rtc::scoped_refptr<RtpReceiverInterface> receiver,
                  const std::vector<rtc::scoped_refptr<MediaStreamInterface>>&
                      streams) override;
  void OnTrack(
      rtc::scoped_refptr<RtpTransceiverInterface> transceiver) override;
  void OnRemoveTrack(
      rtc::scoped_refptr<RtpReceiverInterface> receiver) override;

 private:
  typedef std::map<MediaStreamInterface*, JavaMediaStream>
      NativeToJavaStreamsMap;
  typedef std::map<MediaStreamTrackInterface*, RtpReceiverInterface*>
      NativeMediaStreamTrackToNativeRtpReceiver;

  // If the NativeToJavaStreamsMap contains the stream, return it.
  // Otherwise, create a new Java MediaStream. Returns a global jobject.
  JavaMediaStream& GetOrCreateJavaStream(
      JNIEnv* env,
      const rtc::scoped_refptr<MediaStreamInterface>& stream);

  // Converts array of streams, creating or re-using Java streams as necessary.
  ScopedJavaLocalRef<jobjectArray> NativeToJavaMediaStreamArray(
      JNIEnv* jni,
      const std::vector<rtc::scoped_refptr<MediaStreamInterface>>& streams);

  const ScopedJavaGlobalRef<jobject> j_observer_global_;

  // C++ -> Java remote streams.
  NativeToJavaStreamsMap remote_streams_;
  std::vector<JavaRtpReceiverGlobalOwner> rtp_receivers_;
  // Holds a reference to the Java transceivers given to the AddTrack
  // callback, so that the shared ownership by the Java object will be
  // properly disposed.
  std::vector<JavaRtpTransceiverGlobalOwner> rtp_transceivers_;
};

// PeerConnection doesn't take ownership of the observer. In Java API, we don't
// want the client to have to manually dispose the observer. To solve this, this
// wrapper class is used for object ownership.
//
// Also stores reference to the deprecated PeerConnection constraints for now.
class OwnedPeerConnection {
 public:
  OwnedPeerConnection(
      rtc::scoped_refptr<PeerConnectionInterface> peer_connection,
      std::unique_ptr<PeerConnectionObserver> observer);
  // Deprecated. PC constraints are deprecated.
  OwnedPeerConnection(
      rtc::scoped_refptr<PeerConnectionInterface> peer_connection,
      std::unique_ptr<PeerConnectionObserver> observer,
      std::unique_ptr<MediaConstraints> constraints);
  ~OwnedPeerConnection();

  PeerConnectionInterface* pc() const { return peer_connection_.get(); }
  const MediaConstraints* constraints() const { return constraints_.get(); }

 private:
  rtc::scoped_refptr<PeerConnectionInterface> peer_connection_;
  std::unique_ptr<PeerConnectionObserver> observer_;
  std::unique_ptr<MediaConstraints> constraints_;
};

}  // namespace jni
}  // namespace webrtc

#endif  // SDK_ANDROID_SRC_JNI_PC_PEER_CONNECTION_H_
