/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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

#if ENABLE(WEBGL)

#include <JavaScriptCore/InspectorProtocolObjects.h>

namespace WebCore {

class InspectorCanvas;
class WebGLProgram;

class InspectorShaderProgram final : public RefCounted<InspectorShaderProgram> {
public:
    static Ref<InspectorShaderProgram> create(WebGLProgram&, InspectorCanvas&);

    const String& identifier() const { return m_identifier; }
    InspectorCanvas& canvas() const { return m_canvas; }
    WebGLProgram& program() const { return m_program; }

    String requestShaderSource(Inspector::Protocol::Canvas::ShaderType);
    bool updateShader(Inspector::Protocol::Canvas::ShaderType, const String& source);

    bool disabled() const { return m_disabled; }
    void setDisabled(bool disabled) { m_disabled = disabled; }

    bool highlighted() const { return m_highlighted; }
    void setHighlighted(bool value) { m_highlighted = value; }

    Ref<Inspector::Protocol::Canvas::ShaderProgram> buildObjectForShaderProgram();

private:
    InspectorShaderProgram(WebGLProgram&, InspectorCanvas&);

    String m_identifier;
    InspectorCanvas& m_canvas;
    WebGLProgram& m_program;
    bool m_disabled { false };
    bool m_highlighted { false };
};

} // namespace WebCore

#endif // ENABLE(WEBGL)
