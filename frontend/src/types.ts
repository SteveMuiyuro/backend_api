 export type Supplier = {
    supplier: string;
    product_name: string;
    price: string;
    location: string;
    product_image_url: string;
    supplier_contact:string;
    email:string;
    website:string;
  }

  export type Message = {
    type: 'user' | 'bot';
    text?: string; // For user text and error messages
    data?: Supplier[]; // For bot responses with supplier data
  };
