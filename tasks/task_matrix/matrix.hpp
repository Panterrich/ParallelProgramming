#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <stdexcept>
#include <stdio.h>
#include <stddef.h>
#include <memory.h>
#include <immintrin.h>
#include <xmmintrin.h>

#define IMPL_3

size_t BLOCK_SIZE = 32;

namespace Matrix
{

template <typename T>
class Matrix
{
private:
    struct ProxyRow; // forward declaration

public:

    explicit Matrix(const size_t nRows,
                    const size_t nCols,
                    const T value = T{}) :
        m_nRows{nRows},
        m_nCols{nCols}
    {
        size_t size = nRows * nCols;

        if (size)
        {
            m_data = new (std::align_val_t(32)) T[size]{};
            std::fill_n(m_data, size, value);
        }
        else
        {
            m_data = nullptr;
        }
    }

    ~Matrix()
    {
        delete [] m_data;
    }

    Matrix(const Matrix& src) :
        m_nRows(src.m_nRows),
        m_nCols(src.m_nCols)
    {
        size_t size = src.m_nRows * src.m_nCols;

        if (size)
        {
            m_data = new T[size]{};
            CopyFrom(src, size);
        }
        else
        {
            m_data = nullptr;
        }
    }

    Matrix& operator=(const Matrix& src)
    {
        if (&src == this) return *this;

        if (m_data                 &&
            m_nRows == src.m_nRows &&
            m_nCols == src.m_nCols)
        {
            CopyFrom(src.data, m_nRows * m_nCols);

            return *this;
        }

        delete [] m_data;

        m_nRows = src.m_nRows;
        m_nCols = src.m_nCols;

        size_t size = src.m_nRows * src.m_nCols;

        if (size)
        {
            m_data = new (std::align_val_t(32)) T[size]{};
            CopyFrom(src, size);
        }
        else
        {
            m_data = nullptr;
        }

        return *this;
    }

    Matrix(Matrix&& src) noexcept :
        m_nRows(src.m_nRows),
        m_nCols(src.m_nCols),
        m_data(src.m_data)
    {
        src.m_data = nullptr;
    }

    Matrix& operator=(Matrix&& src) noexcept
    {
        if (&src == this) return *this;

        delete [] m_data;

        m_nRows = src.m_nRows;
        m_nCols = src.m_nCols;
        m_data  = src.m_data;

        src.m_data = nullptr;

        return *this;
    }

    inline size_t GetNRows() const { return m_nRows; }
    inline size_t GetNCols() const { return m_nCols; }

    inline T& operator()(const size_t row, const size_t col)
    {
        return m_data[m_nCols * row + col];
    }

    inline const T& operator()(const size_t row, const size_t col) const
    {
        return m_data[m_nCols * row + col];
    }

    ProxyRow operator[](const size_t row) const
    {
        return ProxyRow{m_data + m_nCols * row, m_nCols};
    }

    template <typename U>
    friend bool operator==(const Matrix<U>& lhs, const Matrix<U>& rhs);

    static void Strassen(const Matrix& a, const Matrix& b, Matrix& c);

private:

    static const size_t kThreshold = 256;

    Matrix() :
        m_nRows{0},
        m_nCols{0},
        m_data{nullptr}
    {}

    struct ProxyRow
    {
        ProxyRow(T* data, const size_t nColsInRow) noexcept :
            nCols(nColsInRow),
            row(data)
        {}

        T& operator[](const size_t col)
        {
            if (col > nCols)
                throw std::runtime_error("Bad col for getting element");

            return row[col];
        }

        const T& operator[](const size_t col) const
        {
            if (col > nCols)
                throw std::runtime_error("Bad col for getting element");

            return row[col];
        }

        size_t nCols;
        T* row;
    };

    void CopyFrom(const Matrix& src, size_t size) noexcept
    {
        memcpy(m_data, src.m_data, size * sizeof(T));
    }

    void CopyFromExtMatrix(const Matrix& src, size_t xStart, size_t xEnd, size_t yStart, size_t yEnd)
    {
        #pragma omp parallel for collapse(2)
        for (size_t i = yStart; i < yEnd; i++)
            for (size_t j = xStart; j < xEnd; j++)
                operator()(i - yStart, j - xStart) = src(i, j);
    }

    void CopyFromBlock(const Matrix& src, size_t xStart, size_t xEnd, size_t yStart, size_t yEnd)
    {
        #pragma omp parallel for collapse(2)
        for (size_t i = yStart; i < yEnd; i++)
            for (size_t j = xStart; j < xEnd; j++)
                operator()(i, j) = src(i - yStart, j - xStart);
    }

    static size_t ExtendedSize(size_t size)
    {
        size_t result = 0;

        while (size >>= 1) result++;

        return 1ULL << result;
    }

    void GetNewDimension(size_t newSize)
    {
        Matrix extended(newSize, newSize);
        extended.CopyFromBlock(*this, 0, GetNCols(), 0, GetNRows());

        *this = std::move(extended);
    }

    void DivideBlockMatrix(Matrix& a11, Matrix& a12, Matrix& a21, Matrix& a22) const;

    void CollectBlockMatrix(const Matrix& a11, const Matrix& a12, const Matrix& a21, const Matrix& a22);

    static void StrassenBody(const Matrix& a, const Matrix& b, Matrix& c);

    size_t m_nRows;
    size_t m_nCols;

    T* m_data;
};

template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator/(const Matrix<T>& lhs, const Matrix<T>& rhs);

//==============================================================================

template <typename T>
bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    return lhs.GetNCols() == rhs.GetNCols() &&
           lhs.GetNRows() == rhs.GetNRows() &&
           memcmp(lhs.m_data, rhs.m_data, lhs.GetNCols() * rhs.GetNRows() * sizeof(T)) == 0;
}

template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.GetNRows() != rhs.GetNRows() ||
        lhs.GetNCols() != rhs.GetNCols())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.GetNRows(), lhs.GetNCols()};

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < lhs.GetNRows(); i++)
    {
        for (size_t j = 0; j < lhs.GetNCols(); j++)
            res(i, j) = lhs(i, j) + rhs(i, j);
    }

    return res;
}

template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.GetNRows() != rhs.GetNRows() ||
        lhs.GetNCols() != rhs.GetNCols())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.GetNRows(), lhs.GetNCols()};

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < lhs.GetNRows(); i++)
    {
        for (size_t j = 0; j < lhs.GetNCols(); j++)
            res(i, j) = lhs(i, j) - rhs(i, j);
    }

    return res;
}

template <typename T>
Matrix<T> operator/(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.GetNCols() != rhs.GetNRows())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.GetNCols(), rhs.GetNRows()};

    Matrix<T>::Strassen(lhs, rhs, res);

    return res;
}

#ifdef IMPL_3
template <>
Matrix<float> operator*(const Matrix<float>& lhs, const Matrix<float>& rhs)
{
    if (lhs.GetNCols() != rhs.GetNRows())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<float> res{lhs.GetNCols(), rhs.GetNRows()};

    #pragma omp parallel for
    for (size_t bi = 0; bi < lhs.GetNRows(); bi += BLOCK_SIZE)
    {
        for (size_t bj = 0; bj < rhs.GetNCols(); bj += BLOCK_SIZE)
        {
            for (size_t bk = 0; bk < lhs.GetNCols(); bk += BLOCK_SIZE)
            {
                for (size_t i = bi; i < bi + BLOCK_SIZE; i++)
                {
                    for (size_t j = bj; j < bj + BLOCK_SIZE; j += 8)
                    {
                        __m256 resVec = _mm256_load_ps(&res(i, j));

                        for (size_t k = bk; k < bk + BLOCK_SIZE; k++)
                        {
                            __m256 lhsVec = _mm256_set1_ps(lhs(i, k));
                            __m256 rhsVec = _mm256_load_ps(&rhs(k, j));
                            resVec = _mm256_add_ps(resVec, _mm256_mul_ps(lhsVec, rhsVec));
                        }

                        _mm256_store_ps(&res(i, j), resVec);
                    }
                }
            }
        }
    }

    return res;
}
#endif // IMPL_3

#ifdef IMPL_2
template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.GetNCols() != rhs.GetNRows())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.GetNCols(), rhs.GetNRows()};

    #pragma omp parallel for
    for (size_t bi = 0; bi < lhs.GetNRows(); bi += BLOCK_SIZE)
    {
        for (size_t bj = 0; bj < rhs.GetNCols(); bj += BLOCK_SIZE)
        {
            for (size_t bk = 0; bk < lhs.GetNCols(); bk += BLOCK_SIZE)
            {
                for (size_t i = bi; i < bi + BLOCK_SIZE; i++)
                {
                    for (size_t k = bk; k < bk + BLOCK_SIZE; k++)
                    {
                        T value = lhs(i, k);

                        for (size_t j = bj; j < bj + BLOCK_SIZE; j++)
                        {
                            res(i, j) += value * rhs(k, j);
                        }
                    }
                }
            }
        }
    }

    return res;
}
#endif // IMPL_2

#ifdef IMPL_1
template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.GetNCols() != rhs.GetNRows())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.GetNCols(), rhs.GetNRows()};

    #pragma omp parallel for
    for (size_t i = 0; i < lhs.GetNRows(); i++)
    {
        for (size_t k = 0; k < lhs.GetNCols(); k++)
        {
            for (size_t j = 0; j < rhs.GetNCols(); j++)
            {
                res(i, j) += lhs(i, k) * rhs(k, j);
            }
        }
    }

    return res;
}
#endif // IMPL_1

#ifdef IMPL_BASE
template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.GetNCols() != rhs.GetNRows())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.GetNCols(), rhs.GetNRows()};

    #pragma omp parallel for
    for (size_t i = 0; i < lhs.GetNRows(); i++)
    {
        for (size_t j = 0; j < rhs.GetNCols(); j++)
        {
            for (size_t k = 0; k < lhs.GetNCols(); k++)
            {
               res(i, j) += lhs(i, k) * rhs(k, j);
            }
        }
    }

    return res;
}

#endif // IMPL_BASE

template <typename T>
void Matrix<T>::DivideBlockMatrix(Matrix& a11, Matrix& a12, Matrix& a21, Matrix& a22) const
{
    size_t N   = GetNCols();
    size_t N_2 = N >> 1;

    Matrix block11(N_2, N_2);
    Matrix block12(N_2, N_2);
    Matrix block21(N_2, N_2);
    Matrix block22(N_2, N_2);

    block11.CopyFromExtMatrix(*this, 0,   N_2, 0,   N_2);
    block12.CopyFromExtMatrix(*this, N_2, N,   0,   N_2);
    block21.CopyFromExtMatrix(*this, 0,   N_2, N_2, N);
    block22.CopyFromExtMatrix(*this, N_2, N,   N_2, N);

    a11 = std::move(block11);
    a12 = std::move(block12);
    a21 = std::move(block21);
    a22 = std::move(block22);
}

template <typename T>
void Matrix<T>::CollectBlockMatrix(const Matrix& a11, const Matrix& a12, const Matrix& a21, const Matrix& a22)
{
    size_t N_2 = a11.GetNCols();
    size_t N   = N_2 << 1;

    *this = Matrix{N, N};

    CopyFromBlock(a11, 0,   N_2, 0,   N_2);
    CopyFromBlock(a12, N_2, N,   0,   N_2);
    CopyFromBlock(a21, 0,   N_2, N_2, N);
    CopyFromBlock(a22, N_2, N,   N_2, N);
}

template <typename T>
void Matrix<T>::Strassen(const Matrix& a, const Matrix& b, Matrix& c)
{
    if (a.GetNCols() != b.GetNRows())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    size_t newSize = ExtendedSize(std::max(a.GetNCols(), a.GetNRows()));

    if (newSize == a.GetNCols() && newSize == a.GetNRows())
    {
        StrassenBody(a, b, c);
    }
    else
    {
        Matrix aExtended{};
        Matrix bExtended{};
        Matrix cExtended{};

        aExtended.GetNewDimension(newSize);
        bExtended.GetNewDimension(newSize);

        StrassenBody(aExtended, bExtended, cExtended);

        c = Matrix{b.GetNCols(), a.GetNRows()};
        c.CopyFromExtMatrix(cExtended, 0, b.GetNCols(), 0, a.GetNRows());
    }
}

template <typename T>
void Matrix<T>::StrassenBody(const Matrix& a, const Matrix& b, Matrix& c)
{
    size_t N = a.GetNCols();

    if (N <= kThreshold)
    {
        c = a * b;
        return;
    }

    Matrix a11{}, a12{}, a21{}, a22{};
    Matrix b11{}, b12{}, b21{}, b22{};
    Matrix c11{}, c12{}, c21{}, c22{};

    a.DivideBlockMatrix(a11, a12, a21, a22);
    b.DivideBlockMatrix(b11, b12, b21, b22);
    c.DivideBlockMatrix(c11, c12, c21, c22);

    Matrix m1{}, m2{}, m3{}, m4{}, m5{}, m6{}, m7{};

    StrassenBody(a11 + a22, b11 + b22, m1);
    StrassenBody(a21 + a22, b11      , m2);
    StrassenBody(a11      , b12 - b22, m3);
    StrassenBody(      a22, b21 - b11, m4);
    StrassenBody(a11 + a12,       b22, m5);
    StrassenBody(a21 - a11, b11 + b12, m6);
    StrassenBody(a12 - a22, b21 + b22, m7);

    c11 = m1 + m4 - m5 + m7;
    c12 = m3 + m5;
    c21 = m2 + m4;
    c22 = m1 - m2 + m3 + m6;

    c.CollectBlockMatrix(c11, c12, c21, c22);
}

} // namespace Matrix

#endif // MATRIX_HPP
