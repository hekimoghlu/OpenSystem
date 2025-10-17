/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

//===----------------------------------------------------------------------===//
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
//
// This file implements MemorySlot-related interfaces for CIR dialect
// operations.
//
//===----------------------------------------------------------------------===//

#include "language/Core/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;

/// Conditions the deletion of the operation to the removal of all its uses.
static bool forwardToUsers(Operation *op,
                           SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}

//===----------------------------------------------------------------------===//
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

toolchain::SmallVector<MemorySlot> cir::AllocaOp::getPromotableSlots() {
  return {MemorySlot{getResult(), getAllocaType()}};
}

Value cir::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                     OpBuilder &builder) {
  return builder.create<cir::ConstantOp>(getLoc(),
                                         cir::UndefAttr::get(slot.elemType));
}

void cir::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                        BlockArgument argument,
                                        OpBuilder &builder) {}

std::optional<PromotableAllocationOpInterface>
cir::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                       Value defaultValue, OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  this->erase();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Interfaces for LoadOp
//===----------------------------------------------------------------------===//

bool cir::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

bool cir::LoadOp::storesTo(const MemorySlot &slot) { return false; }

Value cir::LoadOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                             Value reachingDef, const DataLayout &dataLayout) {
  toolchain_unreachable("getStored should not be called on LoadOp");
}

bool cir::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getType() == slot.elemType;
}

DeletionKind cir::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  getResult().replaceAllUsesWith(reachingDefinition);
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Interfaces for StoreOp
//===----------------------------------------------------------------------===//

bool cir::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

bool cir::StoreOp::storesTo(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

Value cir::StoreOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                              Value reachingDef, const DataLayout &dataLayout) {
  return getValue();
}

bool cir::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr && slot.elemType == getValue().getType();
}

DeletionKind cir::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Interfaces for CastOp
//===----------------------------------------------------------------------===//

bool cir::CastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (getKind() == cir::CastKind::bitcast)
    return forwardToUsers(*this, newBlockingUses);
  return false;
}

DeletionKind cir::CastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}
