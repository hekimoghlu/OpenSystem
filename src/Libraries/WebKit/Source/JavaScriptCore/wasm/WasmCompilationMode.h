/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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

#include <wtf/Forward.h>

namespace JSC { namespace Wasm {

enum class CompilationMode : uint8_t {
    LLIntMode,
    IPIntMode,
    BBQMode,
    OMGMode,
    OMGForOSREntryMode,
    JSToWasmEntrypointMode,
    JSToWasmICMode,
    WasmToJSMode,
};

constexpr inline bool isAnyInterpreter(CompilationMode compilationMode)
{
    switch (compilationMode) {
    case CompilationMode::LLIntMode:
    case CompilationMode::IPIntMode:
        return true;
    case CompilationMode::BBQMode:
    case CompilationMode::OMGForOSREntryMode:
    case CompilationMode::OMGMode:
    case CompilationMode::JSToWasmEntrypointMode:
    case CompilationMode::JSToWasmICMode:
    case CompilationMode::WasmToJSMode:
        return false;
    }
    RELEASE_ASSERT_NOT_REACHED_UNDER_CONSTEXPR_CONTEXT();
}

constexpr inline bool isAnyBBQ(CompilationMode compilationMode)
{
    switch (compilationMode) {
    case CompilationMode::BBQMode:
        return true;
    case CompilationMode::OMGForOSREntryMode:
    case CompilationMode::LLIntMode:
    case CompilationMode::IPIntMode:
    case CompilationMode::OMGMode:
    case CompilationMode::JSToWasmEntrypointMode:
    case CompilationMode::JSToWasmICMode:
    case CompilationMode::WasmToJSMode:
        return false;
    }
    RELEASE_ASSERT_NOT_REACHED_UNDER_CONSTEXPR_CONTEXT();
}

constexpr inline bool isAnyOMG(CompilationMode compilationMode)
{
    switch (compilationMode) {
    case CompilationMode::OMGMode:
    case CompilationMode::OMGForOSREntryMode:
        return true;
    case CompilationMode::BBQMode:
    case CompilationMode::LLIntMode:
    case CompilationMode::IPIntMode:
    case CompilationMode::JSToWasmEntrypointMode:
    case CompilationMode::JSToWasmICMode:
    case CompilationMode::WasmToJSMode:
        return false;
    }
    RELEASE_ASSERT_NOT_REACHED_UNDER_CONSTEXPR_CONTEXT();
}

constexpr inline bool isAnyWasmToJS(CompilationMode compilationMode)
{
    switch (compilationMode) {
    case CompilationMode::WasmToJSMode:
        return true;
    case CompilationMode::OMGMode:
    case CompilationMode::OMGForOSREntryMode:
    case CompilationMode::BBQMode:
    case CompilationMode::LLIntMode:
    case CompilationMode::IPIntMode:
    case CompilationMode::JSToWasmEntrypointMode:
    case CompilationMode::JSToWasmICMode:
        return false;
    }
    RELEASE_ASSERT_NOT_REACHED_UNDER_CONSTEXPR_CONTEXT();
}

} } // namespace JSC::Wasm
