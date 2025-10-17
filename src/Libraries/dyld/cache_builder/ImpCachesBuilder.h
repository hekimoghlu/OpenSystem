/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#ifndef ImpCachesBuilder_h
#define ImpCachesBuilder_h

#include "Diagnostics.h"
#include "JSONReader.h"
#include <memory>
#include <string_view>
#include <vector>

namespace IMPCaches {
class IMPCachesBuilder;
}


namespace imp_caches
{

struct Dylib;

struct Method
{
    Method(std::string_view name);
    Method() = delete;
    ~Method() = default;
    Method(const Method&) = delete;
    Method& operator=(const Method&) = delete;
    Method(Method&&) = default;
    Method& operator=(Method&&) = default;

    std::string_view name;
};

struct Protocol
{
    Protocol(std::string_view name);
    Protocol() = delete;
    ~Protocol() = default;
    Protocol(const Method&) = delete;
    Protocol& operator=(const Protocol&) = delete;
    Protocol(Protocol&&) = default;
    Protocol& operator=(Protocol&&) = default;

    std::string_view name;
};

struct Property
{
    Property(std::string_view name);
    Property() = delete;
    ~Property() = default;
    Property(const Method&) = delete;
    Property& operator=(const Property&) = delete;
    Property(Property&&) = default;
    Property& operator=(Property&&) = default;

    std::string_view name;
};

struct Class
{
    Class(std::string_view name, bool isMetaClass, bool isRootClass);
    Class() = delete;
    ~Class() = default;
    Class(const Class&) = delete;
    Class& operator=(const Class&) = delete;
    Class(Class&&) = default;
    Class& operator=(Class&&) = default;

    std::string_view    name;
    std::vector<Method> methods;
    bool                isMetaClass     = false;
    bool                isRootClass     = false;
    const Class*        metaClass       = nullptr;
    const Class*        superClass      = nullptr;
    const Dylib*        superClassDylib = nullptr;
};

struct Category
{
    Category(std::string_view name);
    Category() = delete;
    ~Category() = default;
    Category(const Category&) = delete;
    Category& operator=(const Category&) = delete;
    Category(Category&&) = default;
    Category& operator=(Category&&) = default;

    std::string_view        name;
    std::vector<Method>     instanceMethods;
    std::vector<Method>     classMethods;
    std::vector<Protocol>   protocols;
    std::vector<Property>   instanceProperties;
    std::vector<Property>   classProperties;
    const Class*            cls         = nullptr;
    const Dylib*            classDylib  = nullptr;
};

struct Dylib
{
    Dylib(std::string_view installName);
    Dylib() = delete;
    ~Dylib() = default;
    Dylib(const Dylib&) = delete;
    Dylib& operator=(const Dylib&) = delete;
    Dylib(Dylib&&) = default;
    Dylib& operator=(Dylib&&) = default;

    std::string_view        installName;
    std::vector<Class>      classes;
    std::vector<Category>   categories;
};

struct FallbackClass
{
    std::string_view installName;
    std::string_view className;
    bool isMetaClass                = false;

    bool operator==(const FallbackClass& other) const {
        return (isMetaClass == other.isMetaClass)
            && (installName == other.installName)
            && (className == other.className);
    }

    size_t hash() const {
        std::size_t seed = 0;
        seed ^= std::hash<std::string_view>()(installName) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<std::string_view>()(className) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<bool>()(isMetaClass) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }
};

struct FallbackClassHash
{
    size_t operator()(const FallbackClass& value) const
    {
        return value.hash();
    }
};

struct BucketMethod
{
    std::string_view installName;
    std::string_view className;
    std::string_view methodName;
    bool isInstanceMethod;

    bool operator==(const BucketMethod& other) const {
        return isInstanceMethod == other.isInstanceMethod &&
                installName == other.installName &&
                className == other.className &&
                methodName == other.methodName;
    }

    size_t hash() const {
        std::size_t seed = 0;
        seed ^= std::hash<std::string_view>()(installName) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<std::string_view>()(className) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<std::string_view>()(methodName) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        seed ^= std::hash<bool>()(isInstanceMethod) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }
};

struct BucketMethodHash {
    size_t operator()(const BucketMethod& k) const {
        return k.hash();
    }
};

struct Bucket
{
    bool                    isEmptyBucket       = true;
    bool                    isInstanceMethod    = true;
    uint32_t                selOffset;
    std::string_view        installName;
    std::string_view        className;
    std::string_view        methodName;
};

struct IMPCache
{
    // If set, points to the class to fall back to if a lookup on the IMP cache fails.  Otherwise
    // is set to the superclass of this class
    std::optional<FallbackClass> fallback_class;
    uint32_t cache_shift :  5;
    uint32_t cache_mask  : 11;
    uint32_t occupied    : 14;
    uint32_t has_inlines :  1;
    uint32_t padding     :  1;
    uint32_t unused      :  31;
    uint32_t bit_one     :  1;

    std::vector<Bucket> buckets;
};

struct Builder
{
    static const bool verbose = false;

    Builder(const std::vector<Dylib>& dylibs, const json::Node& objcOptimizations);
    ~Builder();
    Builder() = delete;
    Builder(const Builder&) = delete;
    Builder& operator=(const Builder&) = delete;
    Builder(Builder&&) = delete;
    Builder& operator=(Builder&&) = delete;

    void buildImpCaches();

    void forEachSelector(void (^handler)(std::string_view str, uint32_t bufferOffset)) const;
    std::optional<imp_caches::IMPCache> getIMPCache(uint32_t dylibIndex, std::string_view className, bool isMetaClass);

    Diagnostics                     diags;
    TimeRecorder                    time;
    const std::vector<Dylib>&       dylibs;
    const json::Node&               objcOptimizations;

    // Note, we own this pointer, but we can't use a unique pointer without including
    // the header and we want to keep things more separated
    IMPCaches::IMPCachesBuilder*    impCachesBuilder = nullptr;
};

} // namespace imp_caches


#endif /* ImpCachesBuilder_h */
