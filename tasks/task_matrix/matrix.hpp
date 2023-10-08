#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <stdexcept>
#include <stdio.h>
#include <stddef.h>
#include <memory.h>

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
            m_data = new T[size]{};
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

        delete [] m_data;

        m_nRows = src.m_nRows;
        m_nCols = src.m_nCols;

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

    inline size_t getNRows() const { return m_nRows; }
    inline size_t getNCols() const { return m_nCols; }

    ProxyRow operator[](const size_t row) const
    {
        return ProxyRow{m_data + m_nCols * row, m_nCols};
    }

private:

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

    size_t m_nRows;
    size_t m_nCols;

    T* m_data;
};

template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs);

//==============================================================================

template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.getNRows() != rhs.getNRows() ||
        lhs.getNCols() != rhs.getNCols())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.getNRows(), lhs.getNCols()};

    #pragma omp parallel for collapse(1)
    for (size_t i = 0; i < lhs.getNRows(); i++)
    {
        for (size_t j = 0; j < lhs.getNCols(); j++)
            res[i][j] = lhs[i][j] + rhs[i][j];
    }

    return res;
}

// #define TRANSFORM
#ifdef TRANSFORM

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.getNCols() != rhs.getNRows())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.getNCols(), rhs.getNRows()};

    #pragma omp parallel for collapse(1)
    for (size_t i = 0; i < lhs.getNRows(); i++)
    {
        for (size_t k = 0; k < lhs.getNCols(); k++)
        {
            T value = lhs[i][k];

            for (size_t j = 0; j < rhs.getNCols(); j++)
            {
                res[i][j] += value * rhs[k][j];
            }
        }
    }

    return res;
}

#else // TRANSFORM

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
    if (lhs.getNCols() != rhs.getNRows())
        throw std::runtime_error("Bad matrix's sizes for matrix addition");

    Matrix<T> res{lhs.getNCols(), rhs.getNRows()};

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < lhs.getNRows(); i++)
    {
        for (size_t j = 0; j < rhs.getNCols(); j++)
        {
            T value{};

            for (size_t k = 0; k < lhs.getNCols(); k++)
            {
                value += lhs[i][k] * rhs[k][j];
            }

            res[i][j] = value;
        }
    }

    return res;
}

#endif // TRANSFORM


} // namespace Matrix

#endif // MATRIX_HPP
