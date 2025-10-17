/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 6, 2022.
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
#ifndef FloatPoint3D_h
#define FloatPoint3D_h

#include "FloatPoint.h"

namespace WebCore {

class FloatPoint3D {
public:
    FloatPoint3D() = default;

    FloatPoint3D(float x, float y, float z)
        : m_x(x)
        , m_y(y)
        , m_z(z)
    {
    }

    FloatPoint3D(const FloatPoint& p)
        : m_x(p.x())
        , m_y(p.y())
    {
    }

    float x() const { return m_x; }
    void setX(float x) { m_x = x; }

    float y() const { return m_y; }
    void setY(float y) { m_y = y; }
    
    FloatPoint xy() const { return { m_x, m_y }; }
    void setXY(FloatPoint p)
    {
        m_x = p.x();
        m_y = p.y();
    }

    float z() const { return m_z; }
    void setZ(float z) { m_z = z; }
    void set(float x, float y, float z)
    {
        m_x = x;
        m_y = y;
        m_z = z;
    }

    void move(float dx, float dy, float dz = 0)
    {
        m_x += dx;
        m_y += dy;
        m_z += dz;
    }

    void scale(float sx, float sy, float sz)
    {
        m_x *= sx;
        m_y *= sy;
        m_z *= sz;
    }

    bool isZero() const
    {
        return !m_x && !m_y && !m_z;
    }

    void normalize();

    float dot(const FloatPoint3D& a) const
    {
        return m_x * a.x() + m_y * a.y() + m_z * a.z();
    }

    // Sets this FloatPoint3D to the cross product of the passed two.
    // It is safe for "this" to be the same as either or both of the
    // arguments.
    void cross(const FloatPoint3D& a, const FloatPoint3D& b)
    {
        float x = a.y() * b.z() - a.z() * b.y();
        float y = a.z() * b.x() - a.x() * b.z();
        float z = a.x() * b.y() - a.y() * b.x();
        m_x = x;
        m_y = y;
        m_z = z;
    }

    // Convenience function returning "this cross point" as a
    // stack-allocated result.
    FloatPoint3D cross(const FloatPoint3D& point) const
    {
        FloatPoint3D result;
        result.cross(*this, point);
        return result;
    }

    float lengthSquared() const { return this->dot(*this); }
    float length() const { return std::hypot(m_x, m_y, m_z); }
    
    float distanceTo(const FloatPoint3D& a) const;

    friend bool operator==(const FloatPoint3D&, const FloatPoint3D&) = default;

private:
    float m_x { 0 };
    float m_y { 0 };
    float m_z { 0 };
};

inline FloatPoint3D& operator +=(FloatPoint3D& a, const FloatPoint3D& b)
{
    a.move(b.x(), b.y(), b.z());
    return a;
}

inline FloatPoint3D& operator +=(FloatPoint3D& a, const FloatPoint& b)
{
    a.move(b.x(), b.y());
    return a;
}

inline FloatPoint3D& operator -=(FloatPoint3D& a, const FloatPoint3D& b)
{
    a.move(-b.x(), -b.y(), -b.z());
    return a;
}

inline FloatPoint3D& operator -=(FloatPoint3D& a, const FloatPoint& b)
{
    a.move(-b.x(), -b.y());
    return a;
}

inline FloatPoint3D operator+(const FloatPoint3D& a, const FloatPoint3D& b)
{
    return FloatPoint3D(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

inline FloatPoint3D operator+(const FloatPoint3D& a, const FloatPoint& b)
{
    return FloatPoint3D(a.x() + b.x(), a.y() + b.y(), a.z());
}

inline FloatPoint3D operator-(const FloatPoint3D& a, const FloatPoint3D& b)
{
    return FloatPoint3D(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}

inline FloatPoint3D operator-(const FloatPoint3D& a, const FloatPoint& b)
{
    return FloatPoint3D(a.x() - b.x(), a.y() - b.y(), a.z());
}

inline float operator*(const FloatPoint3D& a, const FloatPoint3D& b)
{
    // dot product
    return a.dot(b);
}

inline FloatPoint3D operator*(float k, const FloatPoint3D& v)
{
    return FloatPoint3D(k * v.x(), k * v.y(), k * v.z());
}

inline FloatPoint3D operator*(const FloatPoint3D& v, float k)
{
    return FloatPoint3D(k * v.x(), k * v.y(), k * v.z());
}

inline float FloatPoint3D::distanceTo(const FloatPoint3D& a) const
{
    return (*this - a).length();
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const FloatPoint3D&);

} // namespace WebCore

#endif // FloatPoint3D_h
