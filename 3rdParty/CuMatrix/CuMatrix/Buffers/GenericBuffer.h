#pragma once
#include "cuda_runtime.h"
#include <stdint.h>
enum class CudaDataType : int32_t
{
    //! 32-bit floating point format.
    kFLOAT = 0,

    //! IEEE 16-bit floating-point format.
    kHALF = 1,

    //! 8-bit integer representing a quantized floating-point value.
    kINT8 = 2,

    //! Signed 32-bit integer format.
    kINT32 = 3,

    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    kBOOL = 4,

    //! 64-bit (on x64 system)
    kPOINTER = 5

};


//!
//! \brief  The GenericBuffer class is a templated class for buffers. Attributes to TensorRT samples.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer // address. It must work with nullptr input.
//!
//! 
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    //! 
    GenericBuffer(CudaDataType type = CudaDataType::kFLOAT)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mBuffer(nullptr)
        , mOwnership(true)
    {
    }

    ////!
    ////! \brief Construct a buffer with the specified allocation size in bytes.
    ////!
    //GenericBuffer(size_t size, CudaDataType type)
    //    : mSize(size)
    //    , mCapacity(size)
    //    , mType(type)
    //    , mOwnership(true)
    //{
    //    if (!allocFn(&mBuffer, this->nbBytes()))
    //    {
    //        throw std::bad_alloc();
    //    }
    //}

    void initializeWithSpace(size_t size, CudaDataType type) {
        mSize = size;
        mCapacity = size;
        mType = type;
        mOwnership = true;

        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mType(buf.mType)
        , mBuffer(buf.mBuffer)
        , mOwnership(buf.getOwnerShip())
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = CudaDataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    // takeOwnership will be ignored if pPreAllocBuf is nullptr
    GenericBuffer(size_t size, CudaDataType type, void * pPreAllocBuf = nullptr, bool takeOwnership=false)
        : mSize(size)
        , mCapacity(size)
        , mType(type)
    {
        if (pPreAllocBuf == nullptr)
        {
            initializeWithSpace(size, type);
        }
        else
        {
            mOwnership = takeOwnership;
            mBuffer = pPreAllocBuf;

        }
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeBuf();

            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    static inline uint32_t getElementSize(CudaDataType t) noexcept
    {
        switch (t)
        {
        case CudaDataType::kPOINTER: return 8;
        case CudaDataType::kINT32: return 4;
        case CudaDataType::kFLOAT: return 4;
        case CudaDataType::kHALF: return 2;
        case CudaDataType::kBOOL:
        case CudaDataType::kINT8: return 1;
        }
        return 0;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    size_t size() const
    {
        return mSize;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t nbBytes() const
    {
        return this->size() * getElementSize(mType);
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeBuf();
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;

            mOwnership = true;
        }
    }

    ////!
    ////! \brief Overload of resize that accepts Dims
    ////!
    //void resize(const Dims& dims)
    //{
    //    return this->resize(volume(dims));
    //}

    void freeBuf() {
        if (mOwnership)
        {
            freeFn(mBuffer);
        }
    }

    ~GenericBuffer()
    {
        freeBuf();
    }

    bool getOwnerShip() {
        return mOwnership;
    }

protected:
    size_t mSize{ 0 }, mCapacity{ 0 };
    CudaDataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
    bool mOwnership;

};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class ManagedAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMallocManaged(ptr, size) == cudaSuccess;
    }
};

class ManagedFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        cudaMallocHost(ptr, size);
        //cudaHostAlloc
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        //free(ptr);
        cudaFreeHost(ptr);
    }
};


using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

template <typename AllocFunc, typename FreeFunc, typename Class>
class ClassBuffer
{
public:
    typedef std::shared_ptr<ClassBuffer> SharedPtr;
    typedef ClassBuffer* Ptr;

    ClassBuffer(bool callConstructor = false) {
        if (!allocFn(&data, sizeof(Class)))
        {
            throw std::bad_alloc();
        }

        if (callConstructor)
        {
            constructor();
        }

    }

    ~ClassBuffer()
    {
        freeFn(data);
    }

    void constructor() {
        new(data) Class();
    }

    Class* getData() {
        return (Class *) data;
    }

    Class& operator->() const
    {
        return *target;
    }
protected:
    void* data;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

template<typename Class>
using ManagedClassBuffer = ClassBuffer<ManagedAllocator, ManagedFree, Class>;
template<typename Class>
class DeviceClassBuffer : public ClassBuffer<DeviceAllocator, DeviceFree, Class> 
{
public:

    void fromCPU(Class* pObj) {
        CUDA_CHECK_RET(cudaMemcpy(data, pObj, sizeof(Class), cudaMemcpyHostToDevice));
    }
};
