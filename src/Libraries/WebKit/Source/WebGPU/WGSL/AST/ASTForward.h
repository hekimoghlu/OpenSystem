/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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

namespace WGSL::AST {

class Directive;
class DiagnosticDirective;

class Declaration;
class ConstAssert;

class Attribute;
class AlignAttribute;
class BindingAttribute;
class BuiltinAttribute;
class ConstAttribute;
class DiagnosticAttribute;
class GroupAttribute;
class IdAttribute;
class InterpolateAttribute;
class InvariantAttribute;
class LocationAttribute;
class MustUseAttribute;
class SizeAttribute;
class StageAttribute;
class WorkgroupSizeAttribute;

class Expression;
class AbstractFloatLiteral;
class AbstractIntegerLiteral;
class BinaryExpression;
class BitcastExpression;
class BoolLiteral;
class CallExpression;
class FieldAccessExpression;
class Float32Literal;
class Float16Literal;
class IdentifierExpression;
class IdentityExpression;
class IndexAccessExpression;
class PointerDereferenceExpression;
class Signed32Literal;
class UnaryExpression;
class Unsigned32Literal;

class Function;
class Parameter;

class Identifier;

class Statement;
class AssignmentStatement;
class BreakStatement;
class CallStatement;
class CompoundAssignmentStatement;
class CompoundStatement;
class ConstAssertStatement;
class ContinueStatement;
class DecrementIncrementStatement;
class DiscardStatement;
class ForStatement;
class IfStatement;
class LoopStatement;
class PhonyAssignmentStatement;
class ReturnStatement;
class StaticAssertStatement;
class SwitchStatement;
class VariableStatement;
class WhileStatement;

class Structure;
class StructureMember;

class TypeAlias;

class ArrayTypeExpression;
class ElaboratedTypeExpression;
class ReferenceTypeExpression;

class Variable;
class VariableQualifier;

struct SwitchClause;
struct Diagnostic;
struct Continuing;

enum class BinaryOperation : uint8_t;
enum class ParameterRole : uint8_t;
enum class StructureRole : uint8_t;
enum class UnaryOperation : uint8_t;
enum class VariableFlavor : uint8_t;
enum class VariableRole : uint8_t;

} // namespace WGSL::AST
