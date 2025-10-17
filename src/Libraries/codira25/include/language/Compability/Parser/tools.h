/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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

//===-- language/Compability/Parser/tools.h ----------------------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// Author: Tunjay Akbarli
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_COMPABILITY_PARSER_TOOLS_H_
#define LANGUAGE_COMPABILITY_PARSER_TOOLS_H_

#include "parse-tree.h"

namespace language::Compability::parser {

// GetLastName() isolates and returns a reference to the rightmost Name
// in a variable (i.e., the Name whose symbol's type determines the type
// of the variable or expression).
const Name &GetLastName(const Name &);
const Name &GetLastName(const StructureComponent &);
const Name &GetLastName(const DataRef &);
const Name &GetLastName(const Substring &);
const Name &GetLastName(const Designator &);
const Name &GetLastName(const ProcComponentRef &);
const Name &GetLastName(const ProcedureDesignator &);
const Name &GetLastName(const Call &);
const Name &GetLastName(const FunctionReference &);
const Name &GetLastName(const Variable &);
const Name &GetLastName(const AllocateObject &);

// GetFirstName() isolates and returns a reference to the leftmost Name
// in a variable or entity declaration.
const Name &GetFirstName(const Name &);
const Name &GetFirstName(const StructureComponent &);
const Name &GetFirstName(const DataRef &);
const Name &GetFirstName(const Substring &);
const Name &GetFirstName(const Designator &);
const Name &GetFirstName(const ProcComponentRef &);
const Name &GetFirstName(const ProcedureDesignator &);
const Name &GetFirstName(const Call &);
const Name &GetFirstName(const FunctionReference &);
const Name &GetFirstName(const Variable &);
const Name &GetFirstName(const EntityDecl &);

// When a parse tree node is an instance of a specific type wrapped in
// layers of packaging, return a pointer to that object.
// Implemented with mutually recursive template functions that are
// wrapped in a struct to avoid prototypes.
struct UnwrapperHelper {

  template <typename A, typename B> static const A *Unwrap(B *p) {
    if (p) {
      return Unwrap<A>(*p);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B, bool COPY>
  static const A *Unwrap(const common::Indirection<B, COPY> &x) {
    return Unwrap<A>(x.value());
  }

  template <typename A, typename... Bs>
  static const A *Unwrap(const std::variant<Bs...> &x) {
    return common::visit([](const auto &y) { return Unwrap<A>(y); }, x);
  }

  template <typename A, std::size_t J = 0, typename... Bs>
  static const A *Unwrap(const std::tuple<Bs...> &x) {
    if constexpr (J < sizeof...(Bs)) {
      if (auto result{Unwrap<A>(std::get<J>(x))}) {
        return result;
      }
      return Unwrap<A, (J + 1)>(x);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B>
  static const A *Unwrap(const std::optional<B> &o) {
    if (o) {
      return Unwrap<A>(*o);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B>
  static const A *Unwrap(const UnlabeledStatement<B> &x) {
    return Unwrap<A>(x.statement);
  }
  template <typename A, typename B>
  static const A *Unwrap(const Statement<B> &x) {
    return Unwrap<A>(x.statement);
  }

  template <typename A, typename B> static const A *Unwrap(B &x) {
    if constexpr (std::is_same_v<std::decay_t<A>, std::decay_t<B>>) {
      return &x;
    } else if constexpr (ConstraintTrait<B>) {
      return Unwrap<A>(x.thing);
    } else if constexpr (WrapperTrait<B>) {
      return Unwrap<A>(x.v);
    } else if constexpr (UnionTrait<B>) {
      return Unwrap<A>(x.u);
    } else {
      return nullptr;
    }
  }
};

template <typename A, typename B> const A *Unwrap(const B &x) {
  return UnwrapperHelper::Unwrap<A>(x);
}
template <typename A, typename B> A *Unwrap(B &x) {
  return const_cast<A *>(Unwrap<A, B>(const_cast<const B &>(x)));
}

// Get the CoindexedNamedObject if the entity is a coindexed object.
const CoindexedNamedObject *GetCoindexedNamedObject(const AllocateObject &);
const CoindexedNamedObject *GetCoindexedNamedObject(const DataRef &);
const CoindexedNamedObject *GetCoindexedNamedObject(const Designator &);
const CoindexedNamedObject *GetCoindexedNamedObject(const Variable &);

// Detects parse tree nodes with "source" members.
template <typename A, typename = int> struct HasSource : std::false_type {};
template <typename A>
struct HasSource<A, decltype(static_cast<void>(A::source), 0)>
    : std::true_type {};

// Detects parse tree nodes with "typedExpr" members.
template <typename A, typename = int> struct HasTypedExpr : std::false_type {};
template <typename A>
struct HasTypedExpr<A, decltype(static_cast<void>(A::typedExpr), 0)>
    : std::true_type {};

// GetSource()

template <bool GET_FIRST> struct GetSourceHelper {

  using Result = std::optional<CharBlock>;

  template <typename A> static Result GetSource(A *p) {
    if (p) {
      return GetSource(*p);
    } else {
      return std::nullopt;
    }
  }
  template <typename A>
  static Result GetSource(const common::Indirection<A> &x) {
    return GetSource(x.value());
  }

  template <typename A, bool COPY>
  static Result GetSource(const common::Indirection<A, COPY> &x) {
    return GetSource(x.value());
  }

  template <typename... As>
  static Result GetSource(const std::variant<As...> &x) {
    return common::visit([](const auto &y) { return GetSource(y); }, x);
  }

  template <std::size_t J = 0, typename... As>
  static Result GetSource(const std::tuple<As...> &x) {
    if constexpr (J < sizeof...(As)) {
      constexpr std::size_t index{GET_FIRST ? J : sizeof...(As) - J - 1};
      if (auto result{GetSource(std::get<index>(x))}) {
        return result;
      }
      return GetSource<(J + 1)>(x);
    } else {
      return {};
    }
  }

  template <typename A> static Result GetSource(const std::optional<A> &o) {
    if (o) {
      return GetSource(*o);
    } else {
      return {};
    }
  }

  template <typename A> static Result GetSource(const std::list<A> &x) {
    if constexpr (GET_FIRST) {
      for (const A &y : x) {
        if (auto result{GetSource(y)}) {
          return result;
        }
      }
    } else {
      for (auto iter{x.rbegin()}; iter != x.rend(); ++iter) {
        if (auto result{GetSource(*iter)}) {
          return result;
        }
      }
    }
    return {};
  }

  template <typename A> static Result GetSource(const std::vector<A> &x) {
    if constexpr (GET_FIRST) {
      for (const A &y : x) {
        if (auto result{GetSource(y)}) {
          return result;
        }
      }
    } else {
      for (auto iter{x.rbegin()}; iter != x.rend(); ++iter) {
        if (auto result{GetSource(*iter)}) {
          return result;
        }
      }
    }
    return {};
  }

  template <typename A> static Result GetSource(A &x) {
    if constexpr (HasSource<A>::value) {
      return x.source;
    } else if constexpr (ConstraintTrait<A>) {
      return GetSource(x.thing);
    } else if constexpr (WrapperTrait<A>) {
      return GetSource(x.v);
    } else if constexpr (UnionTrait<A>) {
      return GetSource(x.u);
    } else if constexpr (TupleTrait<A>) {
      return GetSource(x.t);
    } else {
      return {};
    }
  }
};

template <typename A> std::optional<CharBlock> GetSource(const A &x) {
  return GetSourceHelper<true>::GetSource(x);
}
template <typename A> std::optional<CharBlock> GetSource(A &x) {
  return GetSourceHelper<true>::GetSource(const_cast<const A &>(x));
}

template <typename A> std::optional<CharBlock> GetLastSource(const A &x) {
  return GetSourceHelper<false>::GetSource(x);
}
template <typename A> std::optional<CharBlock> GetLastSource(A &x) {
  return GetSourceHelper<false>::GetSource(const_cast<const A &>(x));
}

// Checks whether the assignment statement has a single variable on the RHS.
bool CheckForSingleVariableOnRHS(const AssignmentStmt &);

} // namespace language::Compability::parser
#endif // FORTRAN_PARSER_TOOLS_H_
