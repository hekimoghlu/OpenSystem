/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#pragma once

#include "CanvasRenderingContext2DBase.h"
#include "WebGL2RenderingContext.h"
#include "WebGLRenderingContextBase.h"
#include <JavaScriptCore/TypedArrays.h>
#include <initializer_list>
#include <wtf/JSONValues.h>
#include <wtf/Ref.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class ArrayBuffer;
class ArrayBufferView;

} // namespace JSC

namespace WebCore {

class CanvasGradient;
class CanvasPattern;
class CanvasRenderingContext;
class Element;
class HTMLCanvasElement;
class HTMLImageElement;
class HTMLVideoElement;
class ImageBitmap;
class ImageData;
class OffscreenCanvas;
class Path2D;
class CSSStyleImageValue;
class WebGLBuffer;
class WebGLFramebuffer;
class WebGLProgram;
class WebGLQuery;
class WebGLRenderbuffer;
class WebGLSampler;
class WebGLShader;
class WebGLSync;
class WebGLTexture;
class WebGLTransformFeedback;
class WebGLUniformLocation;
class WebGLVertexArrayObject;
struct DOMMatrix2DInit;
struct DOMPointInit;
struct ImageDataSettings;
enum class RecordingSwizzleType : int;
enum class CanvasDirection;
enum class CanvasFillRule;
enum class CanvasLineCap;
enum class CanvasLineJoin;
enum class CanvasTextAlign;
enum class CanvasTextBaseline;
enum class PredefinedColorSpace : uint8_t;
enum ImageSmoothingQuality;

#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_CSS_TYPED_OM_ARGUMENT(macro) \
    macro(RefPtr<CSSStyleImageValue>&) \
// end of FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_CSS_TYPED_OM_ARGUMENT

#if ENABLE(OFFSCREEN_CANVAS)
#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_OFFSCREEN_CANVAS_ARGUMENT(macro) \
    macro(RefPtr<OffscreenCanvas>&) \
// end of FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_OFFSCREEN_CANVAS_ARGUMENT
#else
#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_OFFSCREEN_CANVAS_ARGUMENT(macro)
#endif // ENABLE(OFFSCREEN_CANVAS)

#if ENABLE(VIDEO)
#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_VIDEO_ARGUMENT(macro) \
    macro(RefPtr<HTMLVideoElement>&) \
// end of FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_VIDEO_ARGUMENT
#else
#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_VIDEO_ARGUMENT(macro)
#endif // ENABLE(VIDEO)

#if ENABLE(WEB_CODECS)
#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_VIDEOFRAME_ARGUMENT(macro) \
    macro(RefPtr<WebCodecsVideoFrame>&) \
// end of FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_VIDEOFRAME_ARGUMENT
#else
#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_VIDEOFRAME_ARGUMENT(macro)
#endif // ENABLE(WEB_CODECS)

#if ENABLE(WEBGL)
#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_WEBGL_ARGUMENT(macro) \
    macro(std::optional<WebGLRenderingContextBase::BufferDataSource>&) \
    macro(std::optional<WebGLRenderingContextBase::TexImageSource>&) \
    macro(WebGLBuffer*) \
    macro(WebGLFramebuffer*) \
    macro(WebGLProgram*) \
    macro(WebGLQuery*) \
    macro(WebGLRenderbuffer*) \
    macro(WebGLRenderingContextBase::BufferDataSource&) \
    macro(WebGLRenderingContextBase::Float32List::VariantType&) \
    macro(WebGLRenderingContextBase::Int32List::VariantType&) \
    macro(WebGLRenderingContextBase::TexImageSource&) \
    macro(WebGLSampler*) \
    macro(WebGLShader*) \
    macro(WebGLSync*) \
    macro(WebGLTexture*) \
    macro(WebGLTransformFeedback*) \
    macro(WebGLUniformLocation*) \
    macro(WebGLVertexArrayObject*) \
    macro(WebGL2RenderingContext::Uint32List::VariantType&) \
// end of FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_WEBGL_ARGUMENT
#else
#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_WEBGL_ARGUMENT(macro)
#endif // ENABLE(WEBGL)

#define FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_ARGUMENT(macro) \
    macro(CanvasDirection) \
    macro(CanvasFillRule) \
    macro(CanvasImageSource&) \
    macro(CanvasLineCap) \
    macro(CanvasLineJoin) \
    macro(CanvasPath::RadiusVariant&) \
    macro(CanvasRenderingContext2DBase::StyleVariant&) \
    macro(CanvasTextAlign) \
    macro(CanvasTextBaseline) \
    macro(DOMMatrix2DInit&) \
    macro(Element*) \
    macro(HTMLImageElement*) \
    macro(ImageBitmap*) \
    macro(ImageData*) \
    macro(ImageDataSettings&) \
    macro(ImageSmoothingQuality) \
    macro(std::optional<float>&) \
    macro(std::optional<double>&) \
    macro(Path2D*) \
    macro(PredefinedColorSpace) \
    macro(RefPtr<CanvasGradient>&) \
    macro(RefPtr<CanvasPattern>&) \
    macro(RefPtr<HTMLCanvasElement>&) \
    macro(RefPtr<HTMLImageElement>&) \
    macro(RefPtr<ImageBitmap>&) \
    macro(RefPtr<ImageData>&) \
    macro(Ref<JSC::ArrayBuffer>&) \
    macro(Ref<JSC::ArrayBufferView>&) \
    macro(RefPtr<JSC::ArrayBuffer>&) \
    macro(RefPtr<JSC::ArrayBufferView>&) \
    macro(RefPtr<JSC::Float32Array>&) \
    macro(RefPtr<JSC::Int32Array>&) \
    macro(RefPtr<JSC::Uint32Array>&) \
    macro(String&) \
    macro(Vector<String>&) \
    macro(Vector<float>&) \
    macro(Vector<double>&) \
    macro(Vector<uint32_t>&) \
    macro(Vector<int32_t>&) \
    macro(Vector<CanvasPath::RadiusVariant>&) \
    macro(double) \
    macro(float) \
    macro(uint64_t) \
    macro(int64_t) \
    macro(uint32_t) \
    macro(int32_t) \
    macro(uint8_t) \
    macro(bool) \
    FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_CSS_TYPED_OM_ARGUMENT(macro) \
    FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_OFFSCREEN_CANVAS_ARGUMENT(macro) \
    FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_VIDEO_ARGUMENT(macro) \
    FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_WEBGL_ARGUMENT(macro) \
    FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_VIDEOFRAME_ARGUMENT(macro) \
// end of FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_ARGUMENT

class InspectorCanvasCallTracer {
public:
    struct ProcessedArgument {
        Ref<JSON::Value> value;
        RecordingSwizzleType swizzleType;
    };

    using ProcessedArguments = std::initializer_list<std::optional<ProcessedArgument>>;

#define PROCESS_ARGUMENT_DECLARATION(ArgumentType) \
    static std::optional<ProcessedArgument> processArgument(CanvasRenderingContext&, ArgumentType); \
// end of PROCESS_ARGUMENT_DECLARATION
    FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_ARGUMENT(PROCESS_ARGUMENT_DECLARATION)
#undef PROCESS_ARGUMENT_DECLARATION

    static void recordAction(CanvasRenderingContext&, String&&, ProcessedArguments&& = { });

    static std::optional<ProcessedArgument> processArgument(const CanvasBase&, uint32_t);
    static void recordAction(const CanvasBase&, String&&, ProcessedArguments&& = { });
};

} // namespace WebCore
