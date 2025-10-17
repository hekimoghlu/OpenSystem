/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
#include "sdk/android/src/jni/pc/media_stream.h"

#include <memory>

#include "sdk/android/generated_peerconnection_jni/MediaStream_jni.h"
#include "sdk/android/native_api/jni/java_types.h"
#include "sdk/android/src/jni/jni_helpers.h"

namespace webrtc {
namespace jni {

JavaMediaStream::JavaMediaStream(
    JNIEnv* env,
    rtc::scoped_refptr<MediaStreamInterface> media_stream)
    : j_media_stream_(
          env,
          Java_MediaStream_Constructor(env,
                                       jlongFromPointer(media_stream.get()))) {
  // Create an observer to update the Java stream when the native stream's set
  // of tracks changes.
  observer_.reset(new MediaStreamObserver(
      media_stream.get(),
      [this](AudioTrackInterface* audio_track,
             MediaStreamInterface* media_stream) {
        OnAudioTrackAddedToStream(audio_track, media_stream);
      },
      [this](AudioTrackInterface* audio_track,
             MediaStreamInterface* media_stream) {
        OnAudioTrackRemovedFromStream(audio_track, media_stream);
      },
      [this](VideoTrackInterface* video_track,
             MediaStreamInterface* media_stream) {
        OnVideoTrackAddedToStream(video_track, media_stream);
      },
      [this](VideoTrackInterface* video_track,
             MediaStreamInterface* media_stream) {
        OnVideoTrackRemovedFromStream(video_track, media_stream);
      }));
  for (rtc::scoped_refptr<AudioTrackInterface> track :
       media_stream->GetAudioTracks()) {
    Java_MediaStream_addNativeAudioTrack(env, j_media_stream_,
                                         jlongFromPointer(track.release()));
  }
  for (rtc::scoped_refptr<VideoTrackInterface> track :
       media_stream->GetVideoTracks()) {
    Java_MediaStream_addNativeVideoTrack(env, j_media_stream_,
                                         jlongFromPointer(track.release()));
  }
  // `j_media_stream` holds one reference. Corresponding Release() is in
  // MediaStream_free, triggered by MediaStream.dispose().
  media_stream.release();
}

JavaMediaStream::~JavaMediaStream() {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  // Remove the observer first, so it doesn't react to events during deletion.
  observer_ = nullptr;
  Java_MediaStream_dispose(env, j_media_stream_);
}

void JavaMediaStream::OnAudioTrackAddedToStream(AudioTrackInterface* track,
                                                MediaStreamInterface* stream) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedLocalRefFrame local_ref_frame(env);
  track->AddRef();
  Java_MediaStream_addNativeAudioTrack(env, j_media_stream_,
                                       jlongFromPointer(track));
}

void JavaMediaStream::OnVideoTrackAddedToStream(VideoTrackInterface* track,
                                                MediaStreamInterface* stream) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedLocalRefFrame local_ref_frame(env);
  track->AddRef();
  Java_MediaStream_addNativeVideoTrack(env, j_media_stream_,
                                       jlongFromPointer(track));
}

void JavaMediaStream::OnAudioTrackRemovedFromStream(
    AudioTrackInterface* track,
    MediaStreamInterface* stream) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedLocalRefFrame local_ref_frame(env);
  Java_MediaStream_removeAudioTrack(env, j_media_stream_,
                                    jlongFromPointer(track));
}

void JavaMediaStream::OnVideoTrackRemovedFromStream(
    VideoTrackInterface* track,
    MediaStreamInterface* stream) {
  JNIEnv* env = AttachCurrentThreadIfNeeded();
  ScopedLocalRefFrame local_ref_frame(env);
  Java_MediaStream_removeVideoTrack(env, j_media_stream_,
                                    jlongFromPointer(track));
}

jclass GetMediaStreamClass(JNIEnv* env) {
  return org_webrtc_MediaStream_clazz(env);
}

static jboolean JNI_MediaStream_AddAudioTrackToNativeStream(
    JNIEnv* jni,
    jlong pointer,
    jlong j_audio_track_pointer) {
  return reinterpret_cast<MediaStreamInterface*>(pointer)->AddTrack(
      rtc::scoped_refptr<AudioTrackInterface>(
          reinterpret_cast<AudioTrackInterface*>(j_audio_track_pointer)));
}

static jboolean JNI_MediaStream_AddVideoTrackToNativeStream(
    JNIEnv* jni,
    jlong pointer,
    jlong j_video_track_pointer) {
  return reinterpret_cast<MediaStreamInterface*>(pointer)->AddTrack(
      rtc::scoped_refptr<VideoTrackInterface>(
          reinterpret_cast<VideoTrackInterface*>(j_video_track_pointer)));
}

static jboolean JNI_MediaStream_RemoveAudioTrack(JNIEnv* jni,
                                                 jlong pointer,
                                                 jlong j_audio_track_pointer) {
  return reinterpret_cast<MediaStreamInterface*>(pointer)->RemoveTrack(
      rtc::scoped_refptr<AudioTrackInterface>(
          reinterpret_cast<AudioTrackInterface*>(j_audio_track_pointer)));
}

static jboolean JNI_MediaStream_RemoveVideoTrack(JNIEnv* jni,
                                                 jlong pointer,
                                                 jlong j_video_track_pointer) {
  return reinterpret_cast<MediaStreamInterface*>(pointer)->RemoveTrack(
      rtc::scoped_refptr<VideoTrackInterface>(
          reinterpret_cast<VideoTrackInterface*>(j_video_track_pointer)));
}

static ScopedJavaLocalRef<jstring> JNI_MediaStream_GetId(JNIEnv* jni,
                                                         jlong j_p) {
  return NativeToJavaString(jni,
                            reinterpret_cast<MediaStreamInterface*>(j_p)->id());
}

}  // namespace jni
}  // namespace webrtc
