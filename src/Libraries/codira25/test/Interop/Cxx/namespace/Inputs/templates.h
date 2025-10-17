/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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

#ifndef TEST_INTEROP_CXX_NAMESPACE_INPUTS_TEMPLATES_H
#define TEST_INTEROP_CXX_NAMESPACE_INPUTS_TEMPLATES_H

namespace TemplatesNS1 {
template <class T>
const char *basicFunctionTemplate(T)
    __attribute__((language_attr("import_unsafe"))) {
  return "TemplatesNS1::basicFunctionTemplate";
}

template <class> struct BasicClassTemplate {
  const char *basicMember() __attribute__((language_attr("import_unsafe"))) {
    return "TemplatesNS1::BasicClassTemplate::basicMember";
  }
};

using BasicClassTemplateChar = BasicClassTemplate<char>;
} // namespace TemplatesNS1

namespace TemplatesNS1 {
namespace TemplatesNS2 {
template <class T> const char *forwardDeclaredFunctionTemplate(T);
template <class> struct ForwardDeclaredClassTemplate;

template <class T> const char *forwardDeclaredFunctionTemplateOutOfLine(T);
template <class> struct ForwardDeclaredClassTemplateOutOfLine;
} // namespace TemplatesNS2
} // namespace TemplatesNS1

namespace TemplatesNS1 {
template <class T>
const char *TemplatesNS2::forwardDeclaredFunctionTemplate(T)
    __attribute__((language_attr("import_unsafe"))) {
  return "TemplatesNS1::TemplatesNS2::forwardDeclaredFunctionTemplate";
}

template <class> struct TemplatesNS2::ForwardDeclaredClassTemplate {
  const char *basicMember() __attribute__((language_attr("import_unsafe"))) {
    return "TemplatesNS1::TemplatesNS2::ForwardDeclaredClassTemplate::basicMember";
  }
};

using ForwardDeclaredClassTemplateChar =
    TemplatesNS2::ForwardDeclaredClassTemplate<char>;
} // namespace TemplatesNS1

template <class T>
const char *
TemplatesNS1::TemplatesNS2::forwardDeclaredFunctionTemplateOutOfLine(T) {
  return "TemplatesNS1::TemplatesNS2::forwardDeclaredFunctionTemplateOutOfLine";
}

template <class>
struct TemplatesNS1::TemplatesNS2::ForwardDeclaredClassTemplateOutOfLine {
  const char *basicMember() __attribute__((language_attr("import_unsafe"))) {
    return "TemplatesNS1::TemplatesNS2::ForwardDeclaredClassTemplateOutOfLine::"
           "basicMember";
  }
};

using ForwardDeclaredClassTemplateOutOfLineChar =
    TemplatesNS1::TemplatesNS2::ForwardDeclaredClassTemplateOutOfLine<char>;

namespace TemplatesNS1 {
namespace TemplatesNS3 {
template <class> struct BasicClassTemplate {};
} // namespace TemplatesNS3
} // namespace TemplatesNS1

namespace TemplatesNS1 {
namespace TemplatesNS2 {
using BasicClassTemplateChar = TemplatesNS3::BasicClassTemplate<char>;
inline const char *takesClassTemplateFromSibling(BasicClassTemplateChar) {
  return "TemplatesNS1::TemplatesNS2::takesClassTemplateFromSibling";
}
} // namespace TemplatesNS2
} // namespace TemplatesNS1

namespace TemplatesNS4 {
template <class> struct HasSpecialization {};

template <> struct HasSpecialization<int> {};
} // namespace TemplatesNS4

namespace TemplatesNS1 {
using UseTemplate = TemplatesNS4::HasSpecialization<char>;
using UseSpecialized = TemplatesNS4::HasSpecialization<int>;
} // namespace TemplatesNS1

namespace TemplatesNS1 {
template <class T> const char *basicFunctionTemplateDefinedInDefs(T);
template <class> struct BasicClassTemplateDefinedInDefs;
} // namespace TemplatesNS1

#endif // TEST_INTEROP_CXX_NAMESPACE_INPUTS_TEMPLATES_H
