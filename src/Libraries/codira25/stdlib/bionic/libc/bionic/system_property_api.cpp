/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include <sys/system_properties.h>

#include <async_safe/CHECK.h>
#include <system_properties/prop_area.h>
#include <system_properties/system_properties.h>

#include "private/bionic_defs.h"

static SystemProperties system_properties;
static_assert(__is_trivially_constructible(SystemProperties),
              "System Properties must be trivially constructable");

// This is public because it was exposed in the NDK. As of 2017-01, ~60 apps reference this symbol.
// It is set to nullptr and never modified.
__BIONIC_WEAK_VARIABLE_FOR_NATIVE_BRIDGE
prop_area* __system_property_area__ = nullptr;

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_properties_init() {
  return system_properties.Init(PROP_DIRNAME) ? 0 : -1;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_property_set_filename(const char*) {
  return -1;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_property_area_init() {
  bool fsetxattr_fail = false;
  return system_properties.AreaInit(PROP_DIRNAME, &fsetxattr_fail) && !fsetxattr_fail ? 0 : -1;
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
uint32_t __system_property_area_serial() {
  return system_properties.AreaSerial();
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
const prop_info* __system_property_find(const char* name) {
  return system_properties.Find(name);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_property_read(const prop_info* pi, char* name, char* value) {
  return system_properties.Read(pi, name, value);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
void __system_property_read_callback(const prop_info* pi,
                                     void (*callback)(void* cookie, const char* name,
                                                      const char* value, uint32_t serial),
                                     void* cookie) {
  return system_properties.ReadCallback(pi, callback, cookie);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_property_get(const char* name, char* value) {
  return system_properties.Get(name, value);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_property_update(prop_info* pi, const char* value, unsigned int len) {
  return system_properties.Update(pi, value, len);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_property_add(const char* name, unsigned int namelen, const char* value,
                          unsigned int valuelen) {
  return system_properties.Add(name, namelen, value, valuelen);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
uint32_t __system_property_serial(const prop_info* pi) {
  // N.B. a previous version of this function was much heavier-weight
  // and enforced acquire semantics, so give our load here acquire
  // semantics just in case somebody depends on
  // __system_property_serial enforcing memory order, e.g., in case
  // someone spins on the result of this function changing before
  // loading some value.
  return atomic_load_explicit(&pi->serial, memory_order_acquire);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
uint32_t __system_property_wait_any(uint32_t old_serial) {
  return system_properties.WaitAny(old_serial);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
bool __system_property_wait(const prop_info* pi, uint32_t old_serial, uint32_t* new_serial_ptr,
                            const timespec* relative_timeout) {
  return system_properties.Wait(pi, old_serial, new_serial_ptr, relative_timeout);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
const prop_info* __system_property_find_nth(unsigned n) {
  return system_properties.FindNth(n);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_property_foreach(void (*propfn)(const prop_info* pi, void* cookie), void* cookie) {
  return system_properties.Foreach(propfn, cookie);
}

__BIONIC_WEAK_FOR_NATIVE_BRIDGE
int __system_properties_zygote_reload(void) {
  CHECK(getpid() == gettid());
  return system_properties.Reload(false) ? 0 : -1;
}
